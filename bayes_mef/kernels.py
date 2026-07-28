"""Pure, jittable array kernels for multi-exposure image fusion.

Every function here is side-effect free and broadcast-safe over an arbitrary
trailing image rank, so the same code runs under `jit`, `vmap` and sharding.
"""

import jax.numpy as jnp
from jax.scipy.special import gammainc


def label_f(data, threshold):
    """Uncensored mask m(q): 1 where a count is below the censoring threshold."""
    return (data < threshold).astype(data.dtype)


def rates(times, fused):
    """Per-exposure rates, i.e. np.multiply.outer(times, fused)."""
    return times.reshape(times.shape + (1,) * fused.ndim) * fused


def censored_poisson_mean(threshold, rate):
    """Expectation of a Poisson count given it exceeded the censoring threshold.

    Mirrors `rate * poisson(rate).sf(t - 1) / poisson(rate).sf(t)` via the
    identity `sf(k) == gammainc(floor(k) + 1, rate)`. The `floor` is required:
    `set_threshold` yields non-integer thresholds and SciPy floors before
    evaluating the survival function.
    """
    num = gammainc(jnp.floor(threshold - 1.0) + 1.0, rate)
    den = gammainc(jnp.floor(threshold) + 1.0, rate)

    # Both terms underflow deep in the tail; fall back to `rate` exactly as the
    # original implementation does, so results stay comparable.
    safe_den = jnp.where(den == 0, 1.0, den)
    return jnp.where((num == 0) & (den == 0), rate, rate * num / safe_den)


def mle_fuse(data, lab, times, background):
    """Conventional MEF: maximum-likelihood fusion over uncensored counts only."""
    nu = jnp.clip(data - background, 0, None)
    num = jnp.sum(nu * lab, axis=0)
    den = jnp.tensordot(times, lab, axes=1) + 1e-3
    return num / den


def uncensored_e_step(data, lab, times, fused, background):
    """Uncensored counts, rescaled to remove the background contribution."""
    r = rates(times, fused)
    return r / (r + background + 1e-5) * lab * data


def uncensored_m_step(nu, lab, times, alpha, beta):
    return (jnp.sum(nu, axis=0) + alpha) / (jnp.tensordot(times, lab, axes=1) + beta)


def bayes_e_step(data, cens_idx, has_cens, times, fused, background, threshold):
    """Expectation over censored and uncensored counts alike.

    `censored_poisson_mean` is evaluated only at the censored pixels rather than
    across the whole stack. Censoring is typically well under 1% of pixels, and
    `gammainc` dominates the cost of an EM iteration by three orders of
    magnitude, so this is the difference between the loop being gammainc-bound
    and being memory-bound. The censoring mask depends only on `data` and
    `threshold`, both loop invariant, so the indices are built once on the host
    (see `em.censored_indices`).
    """
    r = rates(times, fused)
    rb = r + background + 1e-5

    flat_data = data.reshape(-1)
    mean_c = censored_poisson_mean(threshold, rb.reshape(-1)[cens_idx])

    # `cens_idx` is edge padded, so repeated targets all write the same value.
    # `has_cens` covers images with no censored pixels at all.
    vals = jnp.where(has_cens, mean_c, flat_data[cens_idx])
    nu = flat_data.at[cens_idx].set(vals).reshape(data.shape)
    return nu * (r / rb)


def bayes_m_step(nu, times, alpha, beta):
    return (jnp.sum(nu, axis=0) + alpha) / (jnp.sum(times) + beta)


def flux_estimate(nu, fused, alpha_flux, beta_flux):
    """MAP update of the per-exposure flux factors."""
    per_exposure = jnp.sum(nu, axis=tuple(range(1, nu.ndim)))
    return (alpha_flux + per_exposure) / (beta_flux + jnp.sum(fused))


def censoring_prior(data, threshold, alpha=1e-3, beta=1e-3):
    """Gamma prior whose mean is the threshold wherever any exposure is censored."""
    mask = jnp.sum((data >= threshold).astype(data.dtype), axis=0) > 0
    return (
        jnp.where(mask, 1.0, alpha),
        jnp.where(mask, 1.0 / threshold, beta),
    )


def ptychogram_mean_prior(data, threshold, alpha=1e-3, beta=1e-3):
    """Gamma prior whose mean is the average diffraction pattern.

    `PtychogramMean` deliberately does not invoke the censoring prior, so alpha
    stays at its scalar default and only beta is derived from the data.
    """
    del threshold, beta
    return jnp.asarray(alpha, data.dtype), alpha / jnp.mean(data, axis=0)
