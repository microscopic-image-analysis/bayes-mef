"""Vectorised MEF over ptychographic scan positions.

Scan positions are fused with a single `vmap`ped, jitted EM program rather than
one process per scan. On a multi-device host (see `bayes_mef.set_cpu_devices`)
the scan axis is additionally spread across devices with `pmap`.
"""

import os
import warnings
from functools import partial

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from . import em, kernels
from ._compat import as_array, as_background, to_numpy, warn_if_precision_risk
from .algos import CensoringPrior, Prior, PtychogramMean
from .utils import corr, timeit

# Vectorised equivalents of the Prior classes, so priors are derived per scan
# inside the same jitted program as the EM itself.
_PRIOR_KERNELS = {
    CensoringPrior: kernels.censoring_prior,
    PtychogramMean: kernels.ptychogram_mean_prior,
}

# Live buffers per scan inside the EM step: data, cens, nu, rates, rates+bg, fused.
_LIVE_BUFFERS = 6
_TARGET_CHUNK_BYTES = 512e6


def set_fluxes(stack, threshold, use_uncensored=True):
    """
    Creating fluxes on the fly when exposure times is None,
    especially useful with experimental data due to its variations.
    """
    # stack should have (n_exposures, n_scans, X, Y) or (n_exposures, X, Y)
    stack = np.asarray(stack)
    pty = stack * (stack < threshold) if use_uncensored else stack

    ind = len(pty) // 2 - 1
    axis_last = tuple(range(1, pty.ndim))
    fluxes = (np.sum(pty, axis=axis_last) + 1e-5) / (np.sum(pty[ind]) + 1e-5)

    return list(fluxes)


def set_threshold(stack):
    """sets threshold based on the exposures per scan"""
    # stack is (n_exposures, X, Y)
    stack = np.asarray(stack)
    return stack.max() - np.sqrt(stack.max())


def _default_batch_size(n_scans, per_scan_shape, itemsize):
    per_scan = int(np.prod(per_scan_shape)) * itemsize * _LIVE_BUFFERS
    return int(np.clip(_TARGET_CHUNK_BYTES // max(per_scan, 1), 1, n_scans))


def _pad_axis0(x, multiple):
    """Pad the scan axis so it divides evenly across devices; sliced off after."""
    remainder = -x.shape[0] % multiple
    if remainder == 0:
        return x
    pad = [(0, remainder)] + [(0, 0)] * (x.ndim - 1)
    return jnp.pad(x, pad, mode="edge")


def _map_over_scans(per_scan, mapped, shared, n_scans, batch_size, n_devices):
    """Run `per_scan` over every scan position, chunked and spread over devices.

    `mapped` are arrays batched along the scan axis; `shared` are broadcast to
    every scan. Within a chunk, `vmap` vectorises the scans and -- when more
    than one device is visible -- `pmap` spreads them across devices. `pmap` is
    used rather than explicit shardings because the EM's nested `lax.scan`
    defeats XLA's sharding propagation.
    """
    in_axes = (0,) * len(mapped) + (None,) * len(shared)
    vectorised = jax.vmap(per_scan, in_axes=in_axes)

    if n_devices > 1:
        runner = jax.pmap(vectorised, in_axes=in_axes)
    else:
        runner = jax.jit(vectorised)

    outputs = []
    for start in range(0, n_scans, batch_size):
        chunk = [a[start : start + batch_size] for a in mapped]
        size = chunk[0].shape[0]

        if n_devices > 1:
            # (n_devices, per_device, ...) so pmap maps devices and vmap scans.
            chunk = [_pad_axis0(a, n_devices) for a in chunk]
            chunk = [a.reshape(n_devices, -1, *a.shape[1:]) for a in chunk]

        out = runner(*chunk, *shared)
        if n_devices > 1:
            out = jax.tree.map(lambda a: a.reshape(-1, *a.shape[2:]), out)
        outputs.append(jax.tree.map(lambda a, n=size: to_numpy(a)[:n], out))

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
):
    """Full EM for a single scan position; vmapped over the scan axis."""
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
    """Interface for launching MEF"""

    def __init__(
        self,
        pty_stack,
        background,
        times=None,
        threshold=None,
        prior: type[Prior] | None = None,
        pty_noisefree_stack=None,
        update_fluxes: bool = False,
    ) -> None:
        pty_stack = np.asarray(pty_stack)
        assert pty_stack.ndim == 4, "expected (n_exposures, n_scans, X, Y)"

        self.threshold = set_threshold(pty_stack) if threshold is None else threshold
        self.times = set_fluxes(pty_stack, self.threshold) if times is None else times
        self.prior = prior
        self.pty_noisefree_stack = pty_noisefree_stack
        self.update_fluxes = update_fluxes

        # move the scan axis first: (n_scans, n_exposures, X, Y)
        self.pty_stack = np.moveaxis(pty_stack, 1, 0)

        background = np.zeros(1) if background is None else np.asarray(background)
        self.background = (
            np.moveaxis(background, 1, 0) if background.ndim == 4 else background
        )

    def _prepare(self, ncpus, batch_size):
        data = as_array(self.pty_stack)
        background = as_background(self.background)
        threshold = as_array(self.threshold)
        times = as_array(self.times).reshape(-1)

        warn_if_precision_risk(threshold)

        n_devices = self._device_count(ncpus)
        n_scans = data.shape[0]
        if batch_size is None:
            batch_size = _default_batch_size(
                n_scans, data.shape[1:], data.dtype.itemsize
            )
        # keep chunks a whole multiple of the device count where possible
        batch_size = max(n_devices, batch_size - batch_size % n_devices or batch_size)

        return data, background, threshold, times, n_scans, batch_size, n_devices

    @staticmethod
    def _device_count(ncpus):
        available = jax.device_count()
        if ncpus is None:
            return available
        if ncpus > available:
            warnings.warn(
                f"requested {ncpus} devices but JAX sees {available}. Call "
                "bayes_mef.set_cpu_devices() before importing jax, or set "
                f"XLA_FLAGS=--xla_force_host_platform_device_count={ncpus}, to "
                "spread scan positions across CPU cores.",
                UserWarning,
                stacklevel=3,
            )
        return min(ncpus, available)

    def _priors(self, data, threshold, n_scans):
        """Per-scan gamma prior parameters, batched along the scan axis."""
        if self.prior is None:
            return (
                jnp.full((n_scans,), em.INIT_ALPHA, data.dtype),
                jnp.full((n_scans,), em.INIT_BETA, data.dtype),
            )

        assert issubclass(self.prior, Prior)
        kernel = _PRIOR_KERNELS.get(self.prior)
        if kernel is not None:
            alpha, beta = jax.vmap(kernel, in_axes=(0, None))(data, threshold)
        else:
            # Unknown user subclass: derive it on the host, one scan at a time.
            pairs = []
            for scan in np.asarray(data):
                p = self.prior(scan, float(threshold))
                p()
                pairs.append((p.alpha, p.beta))
            alpha = jnp.stack(
                [jnp.broadcast_to(as_array(a), scan.shape[1:]) for a, _ in pairs]
            )
            beta = jnp.stack(
                [jnp.broadcast_to(as_array(b), scan.shape[1:]) for _, b in pairs]
            )
        return alpha, beta

    @timeit
    def run_mle(self):
        """runs MLE"""
        data = as_array(self.pty_stack)
        background = as_background(self.background)
        threshold = as_array(self.threshold)
        times = as_array(self.times).reshape(-1)

        def per_scan(scan, bg):
            return kernels.mle_fuse(scan, kernels.label_f(scan, threshold), times, bg)

        if background.ndim == data.ndim:
            mapped, shared = (data, background), ()
        else:
            mapped, shared = (data,), (background,)

        pty_mle = _map_over_scans(
            per_scan, mapped, shared, data.shape[0], data.shape[0], 1
        )

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
            ncpus: number of devices to spread scan positions over. Requires
                `bayes_mef.set_cpu_devices()` or `XLA_FLAGS` to expose more than
                one CPU device; it no longer spawns processes.
            batch_size: scan positions fused at once. Defaults to a size that
                keeps roughly 512 MB of intermediates live.
        """
        data, background, threshold, times, n_scans, batch_size, n_devices = (
            self._prepare(ncpus, batch_size)
        )
        alpha, beta = self._priors(data, threshold, n_scans)

        # A per-scan (4-D) background is mapped; a shared (3-D) one is broadcast.
        cens_idx, has_cens = em.censored_indices(data, threshold)
        cens_idx, has_cens = jnp.asarray(cens_idx), jnp.asarray(has_cens)

        mapped = (data, alpha, beta, cens_idx, has_cens)
        if background.ndim == data.ndim:
            mapped, shared = (*mapped, background), ()
        else:
            shared = (background,)

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
            n_devices,
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
                n_devices,
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
        n_devices,
    ):
        per_scan = partial(
            _fuse_one,
            threshold=threshold,
            times=times,
            n_iter=n_iter,
            n_init=n_init,
            update_fluxes=update_fluxes,
            max_flux=as_array(n_iter),
        )
        return _map_over_scans(per_scan, mapped, shared, n_scans, batch_size, n_devices)

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
