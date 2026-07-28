import time
from functools import wraps

import numpy as np
import numpy.typing as npt
from scipy.special import gammaln

ComplexArr = npt.NDArray[np.complexfloating]


def set_fluxes(
    stack, threshold, use_uncensored=True, *, background=None, common_mask=False
):
    """Estimate flux factors (relative exposure) from the data itself.

    Useful when exposures are not accurately known, especially for experimental
    data. Values are relative to one exposure, so the fused image ends up on a
    relative scale.

    The default sums each exposure over its own uncensored pixels. That
    undercounts long exposures, whose brightest pixels are exactly the censored
    ones, so the ratios are compressed towards 1 precisely when censoring is
    heavy. It is an initialiser for `update_fluxes=True`, not an answer.

    Passing `background` and `common_mask=True` fixes both causes: subtracting
    the background stops a background floor dominating short exposures, and
    restricting to pixels uncensored in *every* exposure compares like with
    like. On a simulated ptychogram with true ratios 0.2:0.5:1:2:4:8 the default
    returns 0.6:1.2:1.0:1.1:1.4:1.2, while the two together return
    0.2:0.5:1.0:2.0:4.0:8.1.

    Args:
        stack (np.ndarray): (n_exposures, ...) with exposures along axis 0.
        threshold (float): censoring threshold.
        use_uncensored (bool, optional): ignore censored pixels. Defaults to True.
        background (np.ndarray or scalar, optional): subtracted before summing.
            Must broadcast against `stack`. Defaults to None (not subtracted).
        common_mask (bool, optional): use only pixels uncensored in every
            exposure. Defaults to False.
    """
    stack = np.asarray(stack)

    if common_mask:
        keep = np.all(stack < threshold, axis=0)
    elif use_uncensored:
        keep = stack < threshold
    else:
        keep = True

    pty = stack if background is None else np.clip(stack - background, 0, None)
    pty = pty * keep

    ind = len(pty) // 2 - 1
    axis_last = tuple(range(1, pty.ndim))
    fluxes = (np.sum(pty, axis=axis_last) + 1e-5) / (np.sum(pty[ind]) + 1e-5)

    return list(fluxes)


def set_threshold(stack):
    """sets threshold based on the exposures per scan"""
    # stack is (n_exposures, X, Y)
    stack = np.asarray(stack)
    return stack.max() - np.sqrt(stack.max())


def corr(a, b):
    """Cross-correlation coefficient."""
    return np.corrcoef(np.asarray(a).ravel(), np.asarray(b).ravel())[0, 1]


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


def timeit(method):
    """custom wrapper for estimating time required for a function to execute"""

    @wraps(method)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = method(*args, **kwargs)
        end_time = time.time()
        print(f"{method.__name__} => {(end_time - start_time):.3f} sec")

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

    image = np.asarray(image)

    # make sure new shape is smaller in every index
    img_shape = np.array(image.shape)
    new_shape = np.array(img_shape // zoom_factor, dtype=int)
    assert np.all(img_shape >= new_shape)

    dy, dx = img_shape - new_shape
    sy = slice(dy // 2, img_shape[0] - dy // 2)
    sx = slice(dx // 2, img_shape[1] - dx // 2)

    new_image = image.copy()
    return new_image[sy, sx]
