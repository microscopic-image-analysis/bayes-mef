# implements the EM methods: uncensored EM, Heuristic EM and full EM

import abc
from typing import TypeAlias

import numpy as np
from scipy.stats import poisson

Vector: TypeAlias = list[float]


class ConventionalMEF:
    """
    Considers the uncensored data (ignores overexposure) from the overexposed dataset
    to estimate maximum likelihood (MLE), background is subtracted from this estimate/data.

    Args:
        data (np.ndarray): 3D array with the first index for acquisition
            times i and the remaining for an image M x N. Therefore i x M x N
        threshold (float): censoring threshold
        times (list): acquisition times
        background (np.ndarray or scalar, optional): background with same shape
            as data or a scalar value. Defaults to 0.
    """

    def __init__(
        self,
        data: np.ndarray,
        threshold: float,
        times: list,
        background: np.ndarray = None,
    ) -> None:

        self.data = data
        self.threshold = threshold
        self.times = times
        self.background = background if background is not None else 0
        self.fused_image = np.ones(data.shape[1:])

    @property
    def label(self):
        """
        returns binary mask m(q) that labels
        censored counts as 0 and uncensored counts as 1
        """

        data = self.data.copy()
        mask = data < self.threshold
        return np.array(mask)

    def mle(self):
        """fused estimate based on mle estimate"""

        # subtract background and clip negative values
        nu = np.clip(self.data - self.background, 0, None)

        # fused image using uncensored data
        fused_image = np.sum(nu * self.label, axis=0)
        fused_image /= np.tensordot(self.times, self.label, axes=1) + 1e-3

        self.fused_image[...] = fused_image


class ExpectationMaximization(abc.ABC):
    """Abstract class for ExpectationMaximization with some common implemented methods

    To implement various types of EM algorithms for MEF

    Args:
        data (np.ndarray): 3D array with the first index for acquisition
            times i and the remaining for an image M x N. Therefore i x M x N
        threshold (float): censoring threshold
        times (list): acquisition times
        background (np.ndarray or scalar, optional): background with same shape
            as data or a scalar value. Defaults to 0.
        alpha (float, optional): gamma prior shape. Defaults to 1.
        beta (float, optional): gamma prior rate. Defaults to 1/threshold.
    """

    def __init__(
        self,
        data: np.ndarray,
        threshold: float,
        times: Vector,
        background: np.ndarray = None,
        update_fluxes: bool = False,
        max_flux_iteration: int = 100,
    ) -> None:
        self.data = data
        self.threshold = threshold
        self.times = times
        self.background = background if background is not None else 0
        self.update_fluxes = update_fluxes

        # prior gamma shape and rate
        self.alpha = 1e-3
        self.beta = 1e-3

        # max flux iterations
        self.max_flux_iterations = max_flux_iteration
        self.current_flux_iteration = 0

        if self.update_fluxes:
            self.alpha_flux = np.full_like(self.times, 1.0)
            self.beta_flux = 1 / np.array(self.times)

        # using setter/getter to set a rate (default: `prior_rate`)
        self.fused_image = None
        self.initialize()

    def initialize(self):
        """Initializes the prior rate based on gamma prior"""

        # prior results to the threshold with no data
        prior_rate = self.alpha / self.beta

        # assuming first argument corresponds to acquisition times
        prior_rate = np.full(self.data.shape[1:], prior_rate, dtype=self.data.dtype)

        self.fused_image = prior_rate

    @property
    def label(self):
        """
        returns binary mask m(q) that labels
        censored counts as 0 and uncensored counts as 1
        """

        data = self.data.copy()
        mask = data < self.threshold
        return np.array(mask)

    @property
    def _rates(self):
        """rates are scaled with times and the current fused image in the iteration"""
        rates = np.multiply.outer(self.times, self.fused_image)

        return rates

    def flux_estimate(self, nu):
        """Updates fluxes based on the current estimate of the fused image"""
        fluxes = self.alpha_flux + np.sum(nu, axis=tuple(range(1, nu.ndim)))
        fluxes /= self.beta_flux + np.sum(self.fused_image)

        return fluxes

    @abc.abstractmethod
    def e_step(self):
        """involves estimating the missing data or incorrect data"""
        pass

    @abc.abstractmethod
    def m_step(self, nu):
        """estimating the fused image with new data"""
        pass

    def __next__(self):
        """Single iteration of EM is implemented as an iterator"""

        nu = self.e_step()
        self.m_step(nu)

        # estimate fluxes if true
        if self.update_fluxes:
            # update fluxes only upto `max_flux_iterations`
            while self.current_flux_iteration < self.max_flux_iterations:
                self.times = self.flux_estimate(nu)
                self.current_flux_iteration += 1

                # exit the loop after every iteration
                break

    def run(self, n_iter: int = 200, return_params=False):
        """iteratively recover the fused image"""

        fused = [self.fused_image.copy()]
        times = [np.array(self.times.copy())]
        for _ in range(n_iter):
            next(self)
            if return_params:
                fused.append(self.fused_image.copy())
                if self.update_fluxes:
                    times.append(self.times.copy())

        # return the parameters if needed
        if return_params:
            if self.update_fluxes:
                return fused, times
            else:
                return fused


class UncensoredEM(ExpectationMaximization):
    """Uncensored EM uses only uncensored data, however, it iteratively removes
    background from the data

    Args:
        data (np.ndarray): 3D array with the first index for acquisition
            times i and the remaining for an image M x N. Therefore i x M x N
        threshold (float): censoring threshold
        times (list): acquisition times
        background (np.ndarray or scalar, optional): background with same shape
            as data or a scalar value. Defaults to 0.
        alpha (float, optional): gamma prior shape. Defaults to 1.
        beta (float, optional): gamma prior rate. Defaults to 1/threshold.
    """

    def e_step(self):
        """
        Estimating the e-step that returns the new data
        with background treatment that accepts the fused image
        """

        # background factor
        bk_factor = self._rates / (self._rates + self.background + 1e-5)

        # returns uncensored data with bk factor
        nu = bk_factor * self.label * self.data

        return nu

    def m_step(self, nu):
        """considers the new updated data with background to estimate the rate"""

        # calculate uncensored mean along exposure time axis
        fused_image = np.sum(nu, axis=0) + self.alpha
        fused_image /= np.tensordot(self.times, self.label, axes=1) + self.beta

        # copy to global image
        self.fused_image[...] = fused_image


class BayesianMEF(UncensoredEM):
    """EM uses censored and uncensored data for fusion, moreover considers the
    background treatment aswell

    Args:
        data (np.ndarray): 3D array with the first index for acquisition
            times i and the remaining for an image M x N. Therefore i x M x N
        threshold (float): censoring threshold
        times (list): acquisition times or flux factors
        background (np.ndarray or scalar, optional): background with same shape
            as data or a scalar value. Defaults to 0.
        alpha (float, optional): gamma prior shape. Defaults to 1.
        beta (float, optional): gamma prior rate. Defaults to 1/threshold
        n_init (int, optional): iterations for the initial rate (initialization)
    """

    def __init__(
        self,
        data: np.ndarray,
        threshold: float,
        times: Vector,
        background: np.ndarray = None,
        update_fluxes: bool = False,
        n_iter_fluxes: int = 100,
        n_init: int = 100,
    ) -> None:
        self.n_init = n_init

        # prior gamma shape and rate
        self.alpha = 1e-3
        self.beta = 1e-3

        # super constructor to access remaining attributes
        super().__init__(
            data, threshold, times, background, update_fluxes, n_iter_fluxes
        )

        if self.update_fluxes:
            self.alpha_flux = np.full_like(self.times, 1.0)
            self.beta_flux = 1 / np.array(self.times)

        # counter
        self.current_flux_iteration = 0

        # initializes fused image
        self.fused_image = None
        self.initialize()

    def initialize(self):
        """
        Initialize fused image by running EM iterations using
        uncensored EM
        """
        # initializes `self.fused_image` to the gamma prior
        super().initialize()

        # uncensored EM as initialization for improving
        # speed of convergence of full EM
        for _ in range(self.n_init):
            nu = super().e_step()
            super().m_step(nu)

    @staticmethod
    def censored_poison(threshold, rate):
        """
        expectation under the censored poisson model that
        estimates missing data
        """

        # truncated poisson
        poi = poisson(rate)
        num = poi.sf(threshold - 1)
        denom = poi.sf(threshold)

        # expectation of missing data

        # quick fix: returns rate when num and denom are zero
        # also avoid zero division by replacing denom zero as 1,
        # although the case where num is non-zero and denom zero should
        # never happen.
        nonzero_denom = np.where(denom == 0, 1, denom)
        cens_mean = np.where(
            (num == 0) & (denom == 0), rate, rate * num / nonzero_denom
        )

        # catching nans of censored mean
        if np.any(np.isnan(cens_mean)):
            print(
                f"Warning: NaN for censored mean at" f"{np.where(np.isnan(cens_mean))}"
            )

        return cens_mean

    def e_step(self):
        """gives an expectation for missing censored poisson data"""

        # rates per acquisition
        rates_bg = self._rates + self.background + 1e-5
        threshold = self.threshold

        # check for censored pixels per acquisition
        nu = self.data.copy()
        for i in range(len(self.times)):
            # do nothing for uncensored acquisitions (label=True)
            if np.all(self.label[i]):
                continue

            # current rate with background for censored pixels
            censored = ~self.label[i]
            rate = rates_bg[i, censored]

            # expectation under the censored poisson model for missing data
            nu[i, censored] = self.censored_poison(threshold, rate)

        # scale data with background factor
        nu *= self._rates / rates_bg

        return nu

    def m_step(self, nu):
        """The maximization step will estimate the mean with all the data"""

        # MAP estimate with the new data
        fused_image = np.sum(nu, axis=0) + self.alpha
        fused_image /= np.sum(self.times) + self.beta

        # update fused image
        self.fused_image[...] = fused_image


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

    def __init__(self, data: np.ndarray, threshold: float) -> None:
        self.data = data
        self.threshold = threshold
        self.alpha = 1e-3
        self.beta = 1e-3

    def cens_label(self) -> np.ndarray:
        """
        returns binary mask m(q) that labels
        censored counts as 1 and uncensored counts as 0
        """

        data = self.data.copy()
        mask = data >= self.threshold
        return np.array(mask)

    def __call__(self) -> np.ndarray:
        # check the presence of censoring (nonzero sum)
        sum_label = np.sum(self.cens_label(), axis=0)

        # labels censored counts as 1 and uncensored as 0
        mask = sum_label > 0

        # shape and rate of gamma distribution
        self.alpha = self.alpha * ~mask + mask
        self.beta = self.beta * ~mask + mask / self.threshold


class PtychogramMean(CensoringPrior):
    def __init__(self, data: np.ndarray, threshold: float) -> None:
        """Exponential Gamma Prior which is a mean of the diffraction
        patterns recorded at different exposure times

        Parameters
        ----------
        data (np.ndarray): 3D array with the first index for acquisition
            times i and the remaining for an image M x N. Therefore i x M x N
        threshold : float
        censoring threshold
        """
        super().__init__(data, threshold)

    def __call__(self) -> np.ndarray:
        self.beta = self.alpha / np.mean(self.data, axis=0)
