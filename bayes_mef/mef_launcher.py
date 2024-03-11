# Uses a parallel implementation for Bayesian MEF
import os
from functools import partial

import h5py
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from bayes_mef.algos import BayesianMEF, ConventionalMEF, ExpectationMaximization, Prior
from bayes_mef.utils import corr, timeit


@timeit
def parallel_fusion(
    em_method: ExpectationMaximization,
    pty_stack: np.ndarray,
    background: np.ndarray,
    threshold: float,
    times: list[float],
    n_iter: int,
    cpus: int,
    prior: Prior,
    update_fluxes: bool,
    max_flux_iterations: int,
):
    """running EM algorithm parallely by allocating CPU cores"""

    # assumes the dimensions to be in the order of (scans, exposure, M, M)
    assert pty_stack.ndim == 4

    # run em partial by proving other params
    run_em_partial = partial(
        run_em,
        em_method=em_method,
        threshold=threshold,
        times=times,
        n_iter=n_iter,
        prior=prior,
        update_fluxes=update_fluxes,
        max_flux_iterations=max_flux_iterations,
    )

    print(f"\n Starting {em_method.__name__} ..")

    # stack as list of tuples of exposures of a scan with background
    pty_stack = [pattern for pattern in pty_stack]

    # assumes the same dimension as data `pty_stack` (scans, exposure, M, M)
    if background.ndim == 4:

        # creates a bg stack
        bg_stack = [bg_per_scan for bg_per_scan in background]

        with Parallel(n_jobs=cpus) as parallel:
            fused = parallel(
                delayed(run_em_partial)(patterns_per_scan, bg_per_scan)
                for patterns_per_scan, bg_per_scan in tqdm(
                    zip(pty_stack, bg_stack), total=len(pty_stack)
                )
            )

    # assumes dimension to be (exposures, M, M)
    if background.ndim == 3:
        # creates another partial function that loads same background for every scan
        run_em_partial_same_bg = partial(run_em_partial, bg=background)

        with Parallel(n_jobs=cpus) as parallel:
            fused = parallel(
                delayed(run_em_partial_same_bg)(pattern) for pattern in tqdm(pty_stack)
            )

    # unpack fused images and updated fluxes
    fused_em, times_em = zip(
        *[(fused_per_scan[0], fused_per_scan[1]) for fused_per_scan in fused]
    )

    return np.array(fused_em), np.array(times_em)


def run_em(
    stack,
    bg,
    em_method,
    threshold,
    times,
    n_iter,
    prior,
    update_fluxes,
    max_flux_iterations,
):
    """runs EM on exposures for a given scan"""

    assert stack.ndim == 3
    assert bg.ndim == 3

    # EM launcher
    em = em_method(stack, threshold, times, bg, update_fluxes, max_flux_iterations)

    # if prior exists, extract the values of alpha and beta
    # or fall back to default values.
    if prior is not None:
        assert issubclass(prior, Prior)
        em_prior = prior(stack, threshold)
        em_prior()

        # set the prior values
        em.alpha = em_prior.alpha
        em.beta = em_prior.beta

    em.run(n_iter)
    fused_em = em.fused_image.copy()
    times_em = em.times.copy()

    return fused_em, times_em


def set_fluxes(stack, threshold, use_uncensored=True):
    """
    Creating fluxes on the fly when exposure times is None,
    especially useful with experimental data due to its variations.
    """
    # stack should have (n_exposures, n_scans, X, Y) or (n_exposures, X, Y)

    if use_uncensored:
        uncens_mask = stack < threshold
        uncens_stack = stack * uncens_mask
        pty = uncens_stack.copy()
    else:
        pty = stack.copy()

    # estimate fluxes
    ind = len(pty) // 2 - 1
    axis_last = tuple(range(1, pty.ndim))
    fluxes = (np.sum(pty, axis=axis_last) + 1e-5) / (np.sum(pty[ind]) + 1e-5)

    return list(fluxes)


def set_threshold(stack):
    """sets threshold based on the exposures per scan"""

    # stack is (n_exposures, X, Y)
    return stack.max() - np.sqrt(stack.max())


class LaunchMEF:
    """Interface for launching MEF"""

    def __init__(
        self,
        pty_stack: np.ndarray,
        background: np.ndarray,
        times: list[float] = None,
        threshold: float = None,
        prior: Prior = None,
        pty_noisefree_stack: np.ndarray = None,
        update_fluxes: bool = False,
    ) -> None:

        # initialize params
        self.pty_stack = pty_stack  # (n_exposures, n_scans, X, Y)
        self.background = background
        self.threshold = (
            set_threshold(self.pty_stack) if threshold is None else threshold
        )
        self.times = (
            set_fluxes(self.pty_stack, self.threshold) if times is None else times
        )
        self.prior = prior
        self.pty_noisefree_stack = pty_noisefree_stack
        self.update_fluxes = update_fluxes

        # (n_exposures, n_scans, X, Y)
        assert self.pty_stack.ndim == 4

        # move second dimension to first (n_scans, n_exposures, X, Y)
        self.pty_stack = np.moveaxis(self.pty_stack, 1, 0)

        if self.background.ndim == 4:
            self.background = np.moveaxis(self.background, 1, 0)

    def run_mle(self):
        """runs MLE"""

        if self.background.ndim == 3:
            pty_mle = []
            for patterns in self.pty_stack:
                # Conventional MEF (MLE)
                conv_mle = ConventionalMEF(
                    patterns, self.threshold, self.times, self.background
                )
                conv_mle.mle()
                pty_mle.append(conv_mle.fused_image)

        if self.background.ndim == 4:
            pty_mle = []
            for patterns, bg in zip(self.pty_stack, self.background):
                # Conventional MEF (MLE)
                conv_mle = ConventionalMEF(patterns, self.threshold, self.times, bg)
                conv_mle.mle()
                pty_mle.append(conv_mle.fused_image)

        if self.pty_noisefree_stack is not None:
            # compare with some ground truth (if exists)
            mle_corr = corr(pty_mle[0], self.pty_noisefree_stack[0])
            print(f"Correlation of MLE with noisefree: {mle_corr:.3f}")

        # min and max for the fused data
        print(f"Min MLE: {np.min(pty_mle):.2f}," f" Max MLE: {np.max(pty_mle):.2f}")

        return np.array(pty_mle)

    def run_em(self, n_iter=200, ncpus=6):
        """Runs EM in parallel"""

        # run in parallel
        pty_em, em_fluxes = parallel_fusion(
            BayesianMEF,
            self.pty_stack,
            self.background,
            self.threshold,
            self.times,
            n_iter=n_iter,
            cpus=ncpus,
            prior=self.prior,
            update_fluxes=self.update_fluxes,
            max_flux_iterations=n_iter,
        )

        # running EM again in parallel without updating fluxes and instead using a fixed
        # average of the corrected fluxes
        if self.update_fluxes:
            corrected_mean_fluxes = list(np.mean(em_fluxes, axis=0))
            print("Corrected fluxes: ", corrected_mean_fluxes)

            pty_em, _ = parallel_fusion(
                BayesianMEF,
                self.pty_stack,
                self.background,
                self.threshold,
                corrected_mean_fluxes,
                n_iter=n_iter,
                cpus=ncpus,
                prior=self.prior,
                update_fluxes=False,
                max_flux_iterations=n_iter,
            )

        if self.pty_noisefree_stack is not None:
            # compare with some ground truth (if exists)
            em_corr = corr(pty_em[0], self.pty_noisefree_stack[0])
            print(f"Correlation of EM with noisefree: {em_corr:.3f}")

        # min and max for the fused data
        print(f"Min EM: {pty_em.min():.2f}," f" Max EM: {pty_em.max():.2f}")

        return pty_em, em_fluxes

    def save_result(
        self,
        filepath: str,
        pty_fused: np.ndarray,
        fluxes_em: np.ndarray,
        params: dict,
        pty_key: str,
        save_misc: bool,
    ):
        """saves result as hdf5"""

        if "orientation" not in params:
            params.update({"orientation": 0})

        with h5py.File(filepath, "w") as hf:
            hf.create_dataset(pty_key, data=pty_fused, dtype="f")
            hf.create_dataset("exposures", data=self.times, dtype="f")
            hf.create_dataset("fluxes", data=fluxes_em)

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
            # hf.create_dataset("No", data=(params["No"],), dtype="i")
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
        ncpus_em: int = 6,
        save_misc: bool = False,
        separate_fused_dir: bool = False,
    ):

        # runs and saves MLE result
        if mle_switch:
            if separate_fused_dir:
                filepath_mle = f"{savepath}/mle_fused/mle_{filename}.hdf5"
                os.makedirs(os.path.dirname(filepath_mle), exist_ok=True)
            else:
                filepath_mle = f"{savepath}/mle_{filename}.hdf5"

            if not os.path.exists(filepath_mle):
                pty_mle = self.run_mle()
                mle_fluxes = self.times
                self.save_result(
                    filepath_mle,
                    pty_mle,
                    mle_fluxes,
                    params,
                    pty_key,
                    save_misc,
                )
            else:
                print(f"File {filepath_mle} already exists")

        # runs and saves EM result
        if em_switch:
            if separate_fused_dir:
                filepath_em = f"{savepath}/em_fused/em_{filename}.hdf5"
                os.makedirs(os.path.dirname(filepath_em), exist_ok=True)
            else:
                filepath_em = f"{savepath}/em_{filename}.hdf5"

            if not os.path.exists(filepath_em):
                pty_em, fluxes_em = self.run_em(n_iter_em, ncpus_em)
                self.save_result(
                    filepath_em,
                    pty_em,
                    fluxes_em,
                    params,
                    pty_key,
                    save_misc,
                )
            else:
                print(f"File {filepath_em} already exists")
