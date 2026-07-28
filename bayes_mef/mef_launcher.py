"""Vectorised MEF over ptychographic scan positions.

Scan positions are fused by a single `vmap`ped, jitted EM program, which runs
unchanged on CPU or GPU. Scans are processed in chunks so that device memory
stays bounded regardless of how large the stack is.
"""

import os
import warnings
from functools import partial

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from . import em, kernels
from ._compat import (
    align_background,
    as_array,
    default_dtype,
    to_numpy,
    warn_if_precision_risk,
)
from .algos import CensoringPrior, Prior, PtychogramMean, estimate_fluxes
from .utils import corr, set_fluxes, set_threshold, timeit

# Vectorised equivalents of the Prior classes, so priors are derived per scan
# inside the same jitted program as the EM itself.
_PRIOR_KERNELS = {
    CensoringPrior: kernels.censoring_prior,
    PtychogramMean: kernels.ptychogram_mean_prior,
}

# Re-exported so `from bayes_mef.mef_launcher import set_fluxes, set_threshold`
# keeps working; they now live in utils so algos can use them too.
__all__ = ["LaunchMEF", "set_fluxes", "set_threshold"]

# Live buffers per scan inside the EM step: data, cens, nu, rates, rates+bg, fused.
_LIVE_BUFFERS = 6

# Used when the device does not report its memory (notably CPU).
_FALLBACK_CHUNK_BYTES = 512 << 20

# Share of free device memory a single chunk may occupy. The rest leaves room
# for XLA scratch and for fragmentation between chunks.
_MEMORY_FRACTION = 0.4


def _warn_ignored_ncpus(ncpus):
    """`ncpus` predates the JAX rewrite and no longer does anything."""
    if ncpus is None:
        return
    warnings.warn(
        f"ncpus={ncpus} is ignored. Scan positions are now fused by a single "
        "vectorised program that uses the whole CPU (or the GPU, if one is "
        "available), so there are no worker processes to size. Use batch_size "
        "to control how many scan positions are fused at once if you need to "
        "cap memory.",
        DeprecationWarning,
        stacklevel=3,
    )


def _chunk_budget_bytes():
    """How many bytes one chunk may hold live on the device."""
    try:
        stats = jax.local_devices()[0].memory_stats() or {}
    except Exception:  # noqa: BLE001 - memory_stats is optional per backend
        stats = {}

    limit = stats.get("bytes_limit")
    if not limit:
        return _FALLBACK_CHUNK_BYTES
    free = max(int(limit) - int(stats.get("bytes_in_use", 0)), 0)
    return max(int(free * _MEMORY_FRACTION), _FALLBACK_CHUNK_BYTES)


def _default_batch_size(n_scans, per_scan_shape, itemsize):
    per_scan = int(np.prod(per_scan_shape)) * itemsize * _LIVE_BUFFERS
    return int(np.clip(_chunk_budget_bytes() // max(per_scan, 1), 1, n_scans))


def _map_over_scans(per_scan, mapped, shared, n_scans, batch_size):
    """Run `per_scan` over every scan position, one chunk at a time.

    `mapped` holds (host array, cast) pairs batched along the scan axis;
    `shared` holds device arrays broadcast to every scan. `vmap` vectorises the
    scans within a chunk and `jit` compiles the whole thing once.

    Mapped arrays stay in host memory and only the current chunk is transferred.
    A ptychogram stack is easily tens of GB, so uploading it whole would exhaust
    device memory however small `batch_size` is: chunking has to bound residency,
    not just the width of the computation.
    """
    in_axes = (0,) * len(mapped) + (None,) * len(shared)
    runner = jax.jit(jax.vmap(per_scan, in_axes=in_axes))

    dtype = default_dtype()
    outputs = []
    for start in range(0, n_scans, batch_size):
        chunk = [a[start : start + batch_size] for a, _ in mapped]
        size = chunk[0].shape[0]

        # Transfer (and cast) only this chunk.
        chunk = [
            jnp.asarray(a, dtype) if cast else jnp.asarray(a)
            for a, (_, cast) in zip(chunk, mapped, strict=True)
        ]

        try:
            out = runner(*chunk, *shared)
            outputs.append(jax.tree.map(lambda a, n=size: to_numpy(a)[:n], out))
        except jax.errors.JaxRuntimeError as exc:
            if "RESOURCE_EXHAUSTED" not in str(exc):
                raise
            raise MemoryError(
                f"ran out of device memory fusing {size} scan positions at once. "
                f"Pass a smaller batch_size to run_em() (currently {batch_size}), "
                "or free memory held by other work on the device."
            ) from exc

    return jax.tree.map(lambda *xs: np.concatenate(xs, axis=0), *outputs)


def _fuse_one(
    data,
    alpha,
    beta,
    cens_idx,
    has_cens,
    background,
    *,
    threshold,
    times,
    n_iter,
    n_init,
    update_fluxes,
    max_flux,
    prior_kernel=None,
):
    """Full EM for a single scan position; vmapped over the scan axis."""
    if prior_kernel is not None:
        # Derived here rather than up front, so the prior never materialises a
        # stack-sized array on the device.
        alpha, beta = prior_kernel(data, threshold)

    state = em.init_bayesian_impl(data, threshold, times, background, n_init=n_init)
    state, _ = em.advance_bayesian_impl(
        state,
        data,
        cens_idx,
        has_cens,
        threshold,
        background,
        alpha,
        beta,
        jnp.ones_like(times),
        1.0 / times,
        max_flux,
        n_iter=n_iter,
        update_fluxes=update_fluxes,
    )
    return state.fused, state.times


class LaunchMEF:
    """Interface for launching MEF.

    Args:
        pty_stack (np.ndarray): (n_exposures, n_scans, X, Y) ptychogram.
        background (np.ndarray or scalar, optional): dark counts, in whatever
            form they were recorded -- a scalar, one offset per exposure
            `(n_exposures,)`, a single dark frame `(X, Y)`, one frame per
            exposure `(n_exposures, X, Y)`, or one per image, matching
            `pty_stack`. See `align_background`. Defaults to 0 when None.
        times (list, optional): acquisition times or flux factors, one per
            exposure. Estimated from the data when omitted.
        threshold (float, optional): censoring threshold. Estimated when omitted.
        prior (type[Prior], optional): gamma prior applied per scan position.
        pty_noisefree_stack (np.ndarray, optional): ground truth, for reporting
            the correlation of the fused result against it.
        update_fluxes (bool, optional): re-estimate the flux factors during EM.
            Defaults to refining whenever `times` was estimated (i.e. omitted).
        flux_init (str, optional): initialiser used when `times` is omitted.
    """

    def __init__(
        self,
        pty_stack,
        background=None,
        times=None,
        threshold=None,
        prior: type[Prior] | None = None,
        pty_noisefree_stack=None,
        update_fluxes: bool | None = None,
        flux_init: str = "matched",
    ) -> None:
        pty_stack = np.asarray(pty_stack)
        assert pty_stack.ndim == 4, "expected (n_exposures, n_scans, X, Y)"

        self.threshold = set_threshold(pty_stack) if threshold is None else threshold
        self.prior = prior
        self.pty_noisefree_stack = pty_noisefree_stack
        # Refine the fluxes by default whenever they are estimated (times omitted);
        # trust supplied exposures. Either is overridable.
        self.update_fluxes = (times is None) if update_fluxes is None else update_fluxes

        background = align_background(background, pty_stack.shape, scan_axis=True)

        if times is None:
            # The stack is still (n_exposures, n_scans, X, Y) here, so a shared
            # background with an exposure axis needs a scan axis to broadcast.
            bg = background[:, None] if background.ndim == 3 else background
            times = estimate_fluxes(pty_stack, self.threshold, flux_init, bg)
        self.times = times

        # move the scan axis first: (n_scans, n_exposures, X, Y)
        self.pty_stack = np.moveaxis(pty_stack, 1, 0)

        self.background = (
            np.moveaxis(background, 1, 0) if background.ndim == 4 else background
        )

    def _prepare(self, ncpus, batch_size):
        # `data` and a per-scan `background` stay in host memory; only the
        # current chunk is transferred (see _map_over_scans).
        data = np.asarray(self.pty_stack)
        background = np.asarray(self.background)
        threshold = as_array(self.threshold)
        times = as_array(self.times).reshape(-1)

        warn_if_precision_risk(threshold)

        _warn_ignored_ncpus(ncpus)
        n_scans = data.shape[0]
        if batch_size is None:
            batch_size = _default_batch_size(
                n_scans, data.shape[1:], np.dtype(default_dtype()).itemsize
            )
        return data, background, threshold, times, n_scans, max(int(batch_size), 1)

    def _priors(self, data, threshold, n_scans):
        """Gamma prior parameters for each scan.

        Returns (kernel, alpha, beta). A non-None kernel is applied per scan
        *inside* the fused program, so the prior never materialises a
        stack-sized array on the device; alpha and beta are then just the
        defaults. Unknown Prior subclasses fall back to a host loop, which does
        produce per-scan arrays, but they stay in host memory and are chunked
        with everything else.
        """
        default = (
            np.full(n_scans, em.INIT_ALPHA),
            np.full(n_scans, em.INIT_BETA),
        )
        if self.prior is None:
            return None, *default

        assert issubclass(self.prior, Prior)
        kernel = _PRIOR_KERNELS.get(self.prior)
        if kernel is not None:
            return kernel, *default

        alphas, betas = [], []
        for scan in data:
            p = self.prior(scan, float(threshold))
            p()
            alphas.append(np.broadcast_to(np.asarray(p.alpha), scan.shape[1:]))
            betas.append(np.broadcast_to(np.asarray(p.beta), scan.shape[1:]))
        return None, np.stack(alphas), np.stack(betas)

    @timeit
    def run_mle(self):
        """runs MLE"""
        data, background, threshold, times, n_scans, batch_size = self._prepare(
            None, None
        )

        def per_scan(scan, bg):
            return kernels.mle_fuse(scan, kernels.label_f(scan, threshold), times, bg)

        if background.ndim == data.ndim:
            mapped = ((data, True), (background, True))
            shared = ()
        else:
            mapped = ((data, True),)
            shared = (as_array(background),)

        pty_mle = _map_over_scans(per_scan, mapped, shared, n_scans, batch_size)

        if self.pty_noisefree_stack is not None:
            mle_corr = corr(pty_mle[0], self.pty_noisefree_stack[0])
            print(f"Correlation of MLE with noisefree: {mle_corr:.3f}")

        print(f"Min MLE: {np.min(pty_mle):.2f}, Max MLE: {np.max(pty_mle):.2f}")
        return pty_mle

    @timeit
    def run_em(self, n_iter=200, ncpus=None, *, batch_size=None, n_init=100):
        """Runs EM over all scan positions.

        Args:
            n_iter: EM iterations per scan position.
            ncpus: ignored, kept so existing call sites keep working. Fusion is
                a single vectorised program now, with no worker processes.
            batch_size: scan positions fused at once. Defaults to a size that
                fits the device's free memory; lower it if you hit MemoryError.
        """
        data, background, threshold, times, n_scans, batch_size = self._prepare(
            ncpus, batch_size
        )
        prior_kernel, alpha, beta = self._priors(data, threshold, n_scans)
        cens_idx, has_cens = em.censored_indices(data, threshold)

        mapped = (
            (data, True),
            (alpha, True),
            (beta, True),
            (cens_idx, False),
            (has_cens, False),
        )
        # A per-scan (4-D) background is mapped; a shared (3-D) one is broadcast.
        if background.ndim == data.ndim:
            mapped, shared = (*mapped, (background, True)), ()
        else:
            shared = (as_array(background),)

        pty_em, em_fluxes = self._fuse(
            mapped,
            shared,
            threshold,
            times,
            n_iter,
            n_init,
            self.update_fluxes,
            n_scans,
            batch_size,
            prior_kernel,
        )

        # Re-run with a fixed average of the corrected fluxes.
        if self.update_fluxes:
            corrected = np.mean(em_fluxes, axis=0)
            print("Corrected fluxes: ", list(corrected))
            pty_em, _ = self._fuse(
                mapped,
                shared,
                threshold,
                as_array(corrected),
                n_iter,
                n_init,
                False,
                n_scans,
                batch_size,
                prior_kernel,
            )

        if self.pty_noisefree_stack is not None:
            em_corr = corr(pty_em[0], self.pty_noisefree_stack[0])
            print(f"Correlation of EM with noisefree: {em_corr:.3f}")

        print(f"Min EM: {pty_em.min():.2f}, Max EM: {pty_em.max():.2f}")
        return pty_em, em_fluxes

    def _fuse(
        self,
        mapped,
        shared,
        threshold,
        times,
        n_iter,
        n_init,
        update_fluxes,
        n_scans,
        batch_size,
        prior_kernel=None,
    ):
        per_scan = partial(
            _fuse_one,
            prior_kernel=prior_kernel,
            threshold=threshold,
            times=times,
            n_iter=n_iter,
            n_init=n_init,
            update_fluxes=update_fluxes,
            max_flux=as_array(n_iter),
        )
        return _map_over_scans(per_scan, mapped, shared, n_scans, batch_size)

    def save_result(self, filepath, pty_fused, fluxes_em, params, pty_key, save_misc):
        """saves result as hdf5"""
        if "orientation" not in params:
            params.update({"orientation": 0})

        with h5py.File(filepath, "w") as hf:
            hf.create_dataset(pty_key, data=np.asarray(pty_fused), dtype="f")
            hf.create_dataset("exposures", data=np.asarray(self.times), dtype="f")
            hf.create_dataset("fluxes", data=np.asarray(fluxes_em))

            if save_misc:
                hf.create_dataset("ptychogram_noisy", data=self.pty_stack, dtype="f")
                hf.create_dataset("threshold", data=self.threshold, dtype="f")
                hf.create_dataset("background", data=self.background, dtype="f")

            if self.pty_noisefree_stack is not None:
                hf.create_dataset(
                    "ptychogram_noNoise", data=self.pty_noisefree_stack, dtype="f"
                )

            hf.create_dataset("encoder", data=params["encoder"], dtype="f")
            hf.create_dataset("binningFactor", data=params["binningFactor"], dtype="i")
            hf.create_dataset("dxd", data=(params["dxd"],), dtype="f")
            hf.create_dataset("Nd", data=(params["Nd"],), dtype="i")
            hf.create_dataset("zo", data=(params["zo"],), dtype="f")
            hf.create_dataset("wavelength", data=(params["wavelength"],), dtype="f")
            hf.create_dataset(
                "entrancePupilDiameter",
                data=(params["entrancePupilDiameter"],),
                dtype="f",
            )
            hf.create_dataset("orientation", data=params["orientation"])

        print(f"Saved fused result at {filepath}")

    def run_and_save(
        self,
        savepath: str,
        filename: str,
        params: dict,
        pty_key: str = "ptychogram",
        mle_switch: bool = True,
        em_switch: bool = True,
        n_iter_em: int = 200,
        ncpus_em: int | None = None,
        save_misc: bool = False,
        separate_fused_dir: bool = False,
    ):
        if mle_switch:
            if separate_fused_dir:
                filepath_mle = f"{savepath}/mle_fused/mle_{filename}.hdf5"
                os.makedirs(os.path.dirname(filepath_mle), exist_ok=True)
            else:
                filepath_mle = f"{savepath}/mle_{filename}.hdf5"

            if not os.path.exists(filepath_mle):
                pty_mle = self.run_mle()
                self.save_result(
                    filepath_mle, pty_mle, self.times, params, pty_key, save_misc
                )
            else:
                print(f"File {filepath_mle} already exists")

        if em_switch:
            if separate_fused_dir:
                filepath_em = f"{savepath}/em_fused/em_{filename}.hdf5"
                os.makedirs(os.path.dirname(filepath_em), exist_ok=True)
            else:
                filepath_em = f"{savepath}/em_{filename}.hdf5"

            if not os.path.exists(filepath_em):
                pty_em, fluxes_em = self.run_em(n_iter_em, ncpus_em)
                self.save_result(
                    filepath_em, pty_em, fluxes_em, params, pty_key, save_misc
                )
            else:
                print(f"File {filepath_em} already exists")
