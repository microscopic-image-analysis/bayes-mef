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
