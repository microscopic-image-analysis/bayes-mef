"""NumPy <-> JAX boundary: dtype policy, input coercion, precision control."""

import warnings

import jax
import jax.numpy as jnp
import numpy as np

_X64_WARNED = False


def default_dtype():
    """Working float dtype: float64 when x64 is enabled, float32 otherwise."""
    return jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


def enable_x64():
    """Run bayes_mef in double precision.

    Must be called before any JAX array is created. Equivalent to setting
    ``JAX_ENABLE_X64=1`` in the environment.
    """
    jax.config.update("jax_enable_x64", True)


def as_array(x, dtype=None):
    """Coerce numpy arrays, Python lists/scalars and JAX arrays to a JAX array."""
    return jnp.asarray(x, dtype=default_dtype() if dtype is None else dtype)


def as_background(x):
    """Background may be None, a scalar, or a full array; all broadcast against data."""
    return as_array(0.0 if x is None else x)


def align_background(background, stack_shape, *, scan_axis=False):
    """Lay a recorded background out so it broadcasts against the stack.

    Backgrounds arrive in whatever form the experiment produced: a number, one
    offset per exposure, a single dark frame, one frame per exposure, or one per
    image. All of them describe the same thing, but NumPy lines shapes up from
    the trailing axis, so only some happen to broadcast correctly against a
    stack -- a per-exposure vector silently spreads along the image columns
    instead. Each form is therefore given its exposure axis explicitly here,
    once, so nothing downstream has to reason about the layout again.

    `stack_shape` is the stack as the caller holds it, exposures first:
    (n_exposures, *image), or (n_exposures, n_scans, *image) with
    `scan_axis=True`. Accepted backgrounds, in that same layout:

        None, scalar, (1,)      one offset for the whole stack
        (n_exposures,)          one offset per exposure
        image                   one dark frame, shared by every exposure
        (n_exposures, *image)   one dark frame per exposure
        stack_shape             one dark frame per image (scan_axis only)

    Anything of the right rank whose axes are 1 or the stack's own length is
    accepted too, so a background already written as (n_exposures, 1, 1) or
    (1, X, Y) passes through untouched. Rank is what disambiguates the forms,
    so a shape that is merely broadcastable by NumPy's trailing-axis rule is
    rejected rather than guessed at.

    Returns a NumPy array in the caller's layout. A per-image background that
    does not actually vary across scan positions -- `np.broadcast_to(frame,
    stack.shape)`, the usual way of saying "this frame everywhere" -- collapses
    to the per-exposure form, so the launcher uploads it once rather than
    re-sending a stack-sized copy with every chunk.
    """
    bg = np.asarray(0.0 if background is None else background)
    stack_shape = tuple(stack_shape)
    n_exposures = stack_shape[0]
    image = stack_shape[2:] if scan_axis else stack_shape[1:]

    def fits(target):
        return len(bg.shape) == len(target) and all(
            b in (1, t) for b, t in zip(bg.shape, target, strict=True)
        )

    if bg.ndim == 0 or bg.shape == (1,):
        return bg.reshape(())

    if bg.shape == (n_exposures,):
        return bg.reshape((n_exposures,) + (1,) * len(image))

    if fits(image) or fits((n_exposures, *image)):
        return bg

    if scan_axis and fits(stack_shape):
        # A singleton or zero-strided scan axis means the frame was broadcast,
        # not measured per position, so the axis carries nothing worth keeping.
        if bg.shape[1] == 1 or bg.strides[1] == 0:
            return bg[:, 0]
        return bg

    accepted = [f"a scalar, {(n_exposures,)}, {image}, {(n_exposures, *image)}"]
    if scan_axis:
        accepted.append(f"or {stack_shape}")
    raise ValueError(
        f"background shape {bg.shape} does not fit a stack of shape "
        f"{stack_shape}. Expected {' '.join(accepted)}."
    )


def to_numpy(x):
    """Public results are NumPy, so downstream `.flat`, h5py and in-place idioms work."""
    return np.asarray(x)


# Above this censoring threshold, XLA's float32 igamma loses accuracy near
# rate ~ threshold -- exactly where the EM operates on censored pixels. Measured
# relative error at rate = 0.9 * threshold: 1e-3 at 4096, 2e-2 at 16383, 1e-1 at
# 65279. Below the bound the error stays at or under 2e-3.
X32_SAFE_THRESHOLD = 4096.0


def warn_if_precision_risk(threshold):
    """Warn once when single precision is not accurate enough for this threshold."""
    global _X64_WARNED
    if _X64_WARNED or jax.config.jax_enable_x64:
        return
    if float(np.max(np.asarray(threshold))) <= X32_SAFE_THRESHOLD:
        return
    _X64_WARNED = True
    warnings.warn(
        f"censoring threshold exceeds {X32_SAFE_THRESHOLD:.0f}, where float32 "
        "evaluation of the censored-Poisson expectation loses up to ~10% "
        "accuracy for rates near the threshold. Call bayes_mef.enable_x64() "
        "(or set JAX_ENABLE_X64=1) before creating any array to run in double "
        "precision.",
        UserWarning,
        stacklevel=3,
    )
