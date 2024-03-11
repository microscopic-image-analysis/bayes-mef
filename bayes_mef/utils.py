import time
from functools import wraps

import numpy as np
from scipy.special import gammaln

try:
    from PtyLab import Params
except ImportError:
    print("Need access to fracPy first")

ComplexArr = np.typing.NDArray[np.complexfloating]


def corr(a, b):
    """Cross-correlation coefficient."""
    return np.corrcoef(a.flat, b.flat)[0, 1]


def complex_corr(truth: ComplexArr, recon_image: ComplexArr):
    """Accepts truth and data that are 2D complex-valued reconstructions"""

    # for convenience
    term = lambda img: np.sqrt(np.sum(np.square(np.abs(img))))

    # define num and denom
    num = np.sum(truth.conj() * recon_image)
    denom = term(truth) * term(recon_image)

    return np.abs(num / denom)


def rmse(a, b):
    """root-mean square error"""
    return np.sqrt(np.mean(np.square(a - b)))


def safelog(x, x_min=1e-30):
    """avoids division by zero for log(0)"""
    return np.log(np.clip(x, x_min, None))


def log_factorials(n):
    """log factorials for log of 0, 1!, ..., upto n!"""
    return gammaln(np.arange(1, n + 1))


def default_reconstruction_params(params: object):
    """
    setting some default reconstruction parameters that won't be mostly changed
    """
    assert isinstance(params, Params)

    params.probePowerCorrectionSwitch = True  # probe normalization to measured PSD
    params.modulusEnforcedProbeSwitch = False  # enforce empty beam
    params.comStabilizationSwitch = True  # center of mass stabilization for probe
    params.orthogonalizationSwitch = True  # probe orthogonalization
    params.orthogonalizationFrequency = 10  # probe orthogonalization frequency
    params.fftshiftSwitch = (
        False  # fftswitch for speed probably? Raises error when True
    )
    # standard fluctuation exponential poission
    params.intensityConstraint = "standard"
    params.absorbingProbeBoundary = (
        False  # controls if probe has period boundary conditions
    )
    params.objectContrastSwitch = False  # pushes object to zero outside ROI
    params.absObjectSwitch = False  # force the object to be abs-only
    params.backgroundModeSwitch = False  # background estimate
    params.couplingSwitch = False  # couple adjacent wavelengths
    # couple adjacent wavelengths (relaxation parameter)
    params.couplingAleph = 1
    params.positionCorrectionSwitch = False  # position correction for encoder


def timeit(method):
    """custom wrapper for estimating time required for a function to execute"""

    @wraps(method)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = method(*args, **kwargs)
        end_time = time.time()
        print(f"{method.__name__} => {(end_time-start_time):.3f} sec")

        return result

    return wrapper


def radial_average(image, bins=100, max_radius=None, return_radius=False):
    """Radial average for a given image upto to
    radius = image.shape // 2

    Args:
        image (np.ndarray): 2D image
        bins (int, optional): number of bins to calculate radial average. Defaults to 100.
        return_radius (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """

    # check if square image for simplicity
    M, N = image.shape
    assert M == N

    if max_radius is None:
        max_radius = N / 2

    y, x = np.indices(image.shape)
    distances = np.sqrt((x - max_radius) ** 2 + (y - max_radius) ** 2)
    hist, edges = np.histogram(np.ravel(distances), bins=bins, weights=np.ravel(image))

    r = 0.5 * (edges[:-1] + edges[1:])

    if return_radius:
        return r, hist

    return hist


def psd(image, bins, ft_switch=False):
    """computing Power spectral density of an fft image

    Args:
        image (np.ndarray): Either a normal or fft image
        bins (int): no. of bins to averge radially
        ft_switch (bool): image should be fft, if not compute it.
    """
    if ft_switch:
        image = np.fft.fftshift(np.fft.fftn(image))

    return radial_average(np.square(np.abs(image)), bins)


def crop_image(image, zoom_factor=1.5):
    """just a function to crop an object to focus on the object"""

    # make sure new shape is smaller in every index
    img_shape = np.array(image.shape)
    new_shape = np.array(img_shape // zoom_factor, dtype=int)
    assert np.all(img_shape >= new_shape)

    dy, dx = img_shape - new_shape
    sy = slice(dy // 2, img_shape[0] - dy // 2)
    sx = slice(dx // 2, img_shape[1] - dx // 2)

    new_image = image.copy()
    return new_image[sy, sx]
