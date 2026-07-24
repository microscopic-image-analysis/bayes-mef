import abc
import warnings

import jax.numpy as jnp
import numpy as np

from . import em, kernels
from ._compat import as_array, as_background, to_numpy, warn_if_precision_risk
from .utils import set_fluxes, set_threshold

Vector = list[float]


FLUX_INIT = ("uncensored", "matched")


def estimate_fluxes(data, threshold, flux_init="uncensored", background=None):
    """Flux factors from the data, by either initialiser. See `set_fluxes`."""
    if flux_init not in FLUX_INIT:
        raise ValueError(f"flux_init must be one of {FLUX_INIT}, got {flux_init!r}")
    if flux_init == "uncensored":
        return set_fluxes(data, threshold)
    return set_fluxes(data, threshold, background=background, common_mask=True)


def _resolve(data, threshold, times, background, update_fluxes, flux_init):
    """Fill in the censoring threshold and flux factors from the data.

    Uses the same estimates as `LaunchMEF`, so a single stack does not have to
    be handled differently from a ptychogram. Estimated flux factors are
    relative to one exposure, so the fused image comes out on a relative scale.
    """
    data = np.asarray(data)
    if threshold is None:
        threshold = set_threshold(data)
    if times is None:
        times = estimate_fluxes(data, threshold, flux_init, background)
        _warn_about_estimate(times, update_fluxes)
    return threshold, times


# Below this ratio between the largest and smallest estimated flux factor, the
# estimate carries essentially no information about the exposures.
_DEGENERATE_SPREAD = 1.5


def _warn_about_estimate(times, update_fluxes):
    """Flag estimates that cannot be trusted on their own."""
    t = np.abs(np.asarray(times, dtype=float))
    spread = float(np.max(t) / max(np.min(t), 1e-30))

    if spread < _DEGENERATE_SPREAD:
        warnings.warn(
            f"estimated flux factors span only {spread:.2f}x, so they carry "
            "almost no information about the exposures. Summed counts stop "
            "tracking the flux once the background dominates, which is the "
            "weakly scattering, low SNR regime this method targets. Pass "
            "`times` explicitly if you know the exposures; otherwise "
            "update_fluxes=True recovers part of the range but not all of it.",
            UserWarning,
            stacklevel=4,
        )
    elif not update_fluxes:
        warnings.warn(
            "flux factors were estimated from the data. Estimating them from "
            "summed counts compresses the ratios towards 1, the more so the "
            "heavier the censoring and the weaker the signal against the "
            "background. Pass update_fluxes=True so EM refines them, or "
            "`times` explicitly if you know the exposures.",
            UserWarning,
            stacklevel=4,
        )


class ConventionalMEF:
    """
    Considers the uncensored data (ignores overexposure) from the overexposed dataset
    to estimate maximum likelihood (MLE), background is subtracted from this estimate/data.

    Args:
        data (np.ndarray): 3D array with the first index for acquisition
            times i and the remaining for an image M x N. Therefore i x M x N
        threshold (float, optional): censoring threshold. Estimated from the
            data when omitted.
        times (list, optional): acquisition times or flux factors. Estimated
            from the data when omitted, relative to one exposure.
        background (np.ndarray or scalar, optional): background with same shape
            as data or a scalar value. Defaults to 0.
    """

    def __init__(
        self,
        data,
        threshold=None,
        times=None,
        background=None,
        *,
        flux_init: str = "uncensored",
    ) -> None:
        threshold, times = _resolve(
            data, threshold, times, background, False, flux_init
        )
        self.data = as_array(data)
        self.threshold = as_array(threshold)
        self.times = as_array(times).reshape(-1)
        self.background = as_background(background)
        self._fused = jnp.ones(self.data.shape[1:], dtype=self.data.dtype)

    @property
    def label(self):
        """Binary mask m(q): censored counts 0, uncensored counts 1."""
        return to_numpy(self.data < self.threshold)

    @property
    def fused_image(self):
        return to_numpy(self._fused)

    def mle(self):
        """fused estimate based on mle estimate"""
        self._fused = em.run_mle(self.data, self.threshold, self.times, self.background)


class ExpectationMaximization(abc.ABC):
    """Abstract base for the EM variants used for MEF.

    Args:
        data (np.ndarray): 3D array with the first index for acquisition
            times i and the remaining for an image M x N. Therefore i x M x N
        threshold (float, optional): censoring threshold. Estimated from the
            data when omitted.
        times (list, optional): acquisition times or flux factors. Estimated
            from the data when omitted, relative to one exposure.
        background (np.ndarray or scalar, optional): background with same shape
            as data or a scalar value. Defaults to 0.
        update_fluxes (bool, optional): re-estimate the flux factors each iteration.
        max_flux_iteration (int, optional): stop refreshing fluxes after this many
            iterations.
    """

    def __init__(
        self,
        data,
        threshold=None,
        times=None,
        background=None,
        update_fluxes: bool = False,
        max_flux_iteration: int = 100,
        *,
        flux_init: str = "uncensored",
    ) -> None:
        threshold, times = _resolve(
            data, threshold, times, background, update_fluxes, flux_init
        )
        self.data = as_array(data)
        self.threshold = as_array(threshold)
        self.times = as_array(times).reshape(-1)
        self.background = as_background(background)
        self.update_fluxes = update_fluxes

        # prior gamma shape and rate; the launcher overwrites these after
        # construction, so they are only read at run() time.
        self.alpha = em.INIT_ALPHA
        self.beta = em.INIT_BETA

        self.max_flux_iterations = max_flux_iteration
        self.alpha_flux = jnp.ones_like(self.times)
        self.beta_flux = 1.0 / self.times

        warn_if_precision_risk(self.threshold)
        self._state = self.initialize()

    def initialize(self):
        return em.initial_state(self.data, self.times)

    @property
    def label(self):
        """Binary mask m(q): censored counts 0, uncensored counts 1."""
        return to_numpy(self.data < self.threshold)

    @property
    def fused_image(self):
        return to_numpy(self._state.fused)

    @property
    def times(self):
        return self._times

    @times.setter
    def times(self, value):
        self._times = as_array(value).reshape(-1)
        state = getattr(self, "_state", None)
        if state is not None:
            self._state = state._replace(times=self._times)

    @property
    @abc.abstractmethod
    def _advance(self):
        """Jitted driver advancing the EM state for this variant."""

    def _extra_args(self):
        """Per variant arguments inserted after `data`."""
        return ()

    def _run_iterations(self, n_iter, collect):
        return self._advance(
            self._state,
            self.data,
            *self._extra_args(),
            self.threshold,
            self.background,
            as_array(self.alpha),
            as_array(self.beta),
            self.alpha_flux,
            self.beta_flux,
            as_array(self.max_flux_iterations),
            n_iter=n_iter,
            update_fluxes=self.update_fluxes,
            collect=collect,
        )

    def __next__(self):
        """Single iteration of EM, kept for the original iterator protocol."""
        self._state, _ = self._run_iterations(1, collect=False)

    def run(self, n_iter: int = 200, return_params=False):
        """Iteratively recover the fused image."""
        fused0, times0 = self.fused_image, to_numpy(self._state.times)

        self._state, collected = self._run_iterations(n_iter, collect=return_params)
        self._times = self._state.times

        if np.isnan(self.fused_image).any():
            warnings.warn(
                "fused image contains NaN; check the background and threshold",
                RuntimeWarning,
                stacklevel=2,
            )

        if not return_params:
            return None

        fused_hist, times_hist = collected
        fused = [fused0] + list(to_numpy(fused_hist))
        if self.update_fluxes:
            return fused, [times0] + list(to_numpy(times_hist))
        return fused


class UncensoredEM(ExpectationMaximization):
    """EM over uncensored counts only, iteratively removing the background.

    Args:
        data (np.ndarray): 3D array with the first index for acquisition
            times i and the remaining for an image M x N. Therefore i x M x N
        threshold (float, optional): censoring threshold. Estimated from the
            data when omitted.
        times (list, optional): acquisition times. Estimated from the data when
            omitted, relative to one exposure.
        background (np.ndarray or scalar, optional): background with same shape
            as data or a scalar value. Defaults to 0.
    """

    @property
    def _advance(self):
        return em.advance_uncensored


class BayesianMEF(UncensoredEM):
    """EM over censored and uncensored counts, with background treatment.

    Args:
        data (np.ndarray): 3D array with the first index for acquisition
            times i and the remaining for an image M x N. Therefore i x M x N
        threshold (float, optional): censoring threshold. Estimated from the
            data when omitted.
        times (list, optional): acquisition times or flux factors. Estimated
            from the data when omitted, relative to one exposure.
        background (np.ndarray or scalar, optional): background with same shape
            as data or a scalar value. Defaults to 0.
        update_fluxes (bool, optional): re-estimate the flux factors each iteration.
        n_iter_fluxes (int, optional): stop refreshing fluxes after this many iterations.
        n_init (int, optional): uncensored iterations used to initialise the rate.
    """

    def __init__(
        self,
        data,
        threshold=None,
        times=None,
        background=None,
        update_fluxes: bool = False,
        n_iter_fluxes: int = 100,
        n_init: int = 100,
        *,
        flux_init: str = "uncensored",
    ) -> None:
        self.n_init = n_init
        super().__init__(
            data,
            threshold,
            times,
            background,
            update_fluxes,
            n_iter_fluxes,
            flux_init=flux_init,
        )
        # Censoring is loop invariant, so resolve it once rather than every step.
        self._cens_idx, self._has_cens = em.censored_indices(self.data, self.threshold)

    def _extra_args(self):
        return (self._cens_idx, self._has_cens)

    def initialize(self):
        """Warm start the fused image with uncensored EM to speed up convergence."""
        return em.init_bayesian(
            self.data,
            self.threshold,
            self.times,
            self.background,
            n_init=self.n_init,
        )

    @property
    def _advance(self):
        return em.advance_bayesian


class Prior(abc.ABC):
    @abc.abstractmethod
    def __call__(self):
        pass


class CensoringPrior(Prior):
    """
    A gamma prior whose expected value is an image with censored pixels
    set to the threshold

    Parameters
    ----------
    data (np.ndarray): 3D array with the first index for acquisition
            times i and the remaining for an image M x N. Therefore i x M x N
    threshold : float
        censoring threshold
    """

    def __init__(self, data, threshold=None) -> None:
        threshold = set_threshold(np.asarray(data)) if threshold is None else threshold
        self.data = as_array(data)
        self.threshold = as_array(threshold)
        self.alpha = em.INIT_ALPHA
        self.beta = em.INIT_BETA

    def cens_label(self):
        """Binary mask m(q): censored counts 1, uncensored counts 0."""
        return to_numpy(self.data >= self.threshold)

    def __call__(self):
        self.alpha, self.beta = kernels.censoring_prior(
            self.data, self.threshold, self.alpha, self.beta
        )


class PtychogramMean(CensoringPrior):
    """Gamma prior whose mean is the average of the patterns over exposures.

    Parameters
    ----------
    data (np.ndarray): 3D array with the first index for acquisition
        times i and the remaining for an image M x N. Therefore i x M x N
    threshold : float
        censoring threshold
    """

    def __call__(self):
        self.alpha, self.beta = kernels.ptychogram_mean_prior(
            self.data, self.threshold, self.alpha, self.beta
        )
