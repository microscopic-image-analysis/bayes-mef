"""Jitted EM drivers built on `lax.scan`.

The whole iteration runs on device: one compilation per (shape, dtype, static
config), no Python overhead per step.
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from . import kernels

# Gamma prior in force while initialising, before any user prior is applied.
INIT_ALPHA = 1e-3
INIT_BETA = 1e-3


class EMState(NamedTuple):
    fused: jax.Array
    times: jax.Array
    step: jax.Array


def initial_state(data, times):
    """Fused image starts at the prior mean, alpha / beta."""
    fused = jnp.full(data.shape[1:], INIT_ALPHA / INIT_BETA, dtype=data.dtype)
    return EMState(fused, times, jnp.int32(0))


def _advance(state, e_fn, m_fn, flux, n_iter, collect):
    def step(state, _):
        nu = e_fn(state.times, state.fused)
        fused = m_fn(nu, state.times)

        if flux is None:
            times = state.times
        else:
            alpha_flux, beta_flux, max_flux = flux
            candidate = kernels.flux_estimate(nu, fused, alpha_flux, beta_flux)
            # Fluxes refresh only for the first `max_flux` iterations. The
            # original guards this with a Python counter incremented once per
            # iteration, so the scan index reproduces it exactly. Note the
            # estimate uses `nu` from before the M step but `fused` from after.
            times = jnp.where(state.step < max_flux, candidate, state.times)

        nxt = EMState(fused, times, state.step + 1)
        return nxt, ((fused, times) if collect else None)

    return lax.scan(step, state, None, length=n_iter)


def _uncensored_fns(data, lab, background, alpha, beta):
    def e_fn(times, fused):
        return kernels.uncensored_e_step(data, lab, times, fused, background)

    def m_fn(nu, times):
        return kernels.uncensored_m_step(nu, lab, times, alpha, beta)

    return e_fn, m_fn


def censored_indices(data, threshold):
    """Flat indices of the censored pixels of each image in a stack.

    Host side on purpose: the censoring mask is loop invariant, so resolving it
    once keeps `gammainc` off the ~99% of pixels that are never censored. The
    returned indices are edge padded to a common length so the shape stays static
    under `jit` and `vmap`; repeated targets therefore always scatter the same
    value. `has_cens` handles images with no censored pixels at all.
    """
    data = np.asarray(data)
    t = float(np.asarray(threshold))
    batched = data.ndim == 4

    # One image at a time: comparing the whole stack at once would allocate a
    # boolean array the size of the stack, which for a ptychogram is many GB.
    images = data if batched else data[None]
    found = [np.flatnonzero(np.asarray(img).reshape(-1) >= t) for img in images]
    n_max = max(1, max(len(f) for f in found))

    idx = np.zeros((len(found), n_max), dtype=np.int32)
    has = np.zeros(len(found), dtype=bool)
    for i, f in enumerate(found):
        if len(f):
            idx[i, : len(f)] = f
            idx[i, len(f) :] = f[-1]  # edge pad: duplicate writes agree
            has[i] = True

    return (idx, has) if batched else (idx[0], has[0])


def _bayes_fns(data, cens_idx, has_cens, background, threshold, alpha, beta):
    def e_fn(times, fused):
        return kernels.bayes_e_step(
            data, cens_idx, has_cens, times, fused, background, threshold
        )

    def m_fn(nu, times):
        return kernels.bayes_m_step(nu, times, alpha, beta)

    return e_fn, m_fn


def init_bayesian_impl(data, threshold, times, background, *, n_init):
    """Warm start: `n_init` uncensored iterations under the default gamma prior.

    The prior is deliberately *not* threaded through here. The original runs
    this during construction, before any user prior is assigned, so the warm
    start always sees alpha = beta = 1e-3.
    """
    lab = kernels.label_f(data, threshold)
    e_fn, m_fn = _uncensored_fns(data, lab, background, INIT_ALPHA, INIT_BETA)
    state, _ = _advance(initial_state(data, times), e_fn, m_fn, None, n_init, False)
    # The flux gate counts main-loop iterations, so the warm start does not count.
    return state._replace(step=jnp.int32(0))


def advance_bayesian_impl(
    state,
    data,
    cens_idx,
    has_cens,
    threshold,
    background,
    alpha,
    beta,
    alpha_flux,
    beta_flux,
    max_flux,
    *,
    n_iter,
    update_fluxes=False,
    collect=False,
):
    e_fn, m_fn = _bayes_fns(
        data, cens_idx, has_cens, background, threshold, alpha, beta
    )
    flux = (alpha_flux, beta_flux, max_flux) if update_fluxes else None
    return _advance(state, e_fn, m_fn, flux, n_iter, collect)


def advance_uncensored_impl(
    state,
    data,
    threshold,
    background,
    alpha,
    beta,
    alpha_flux,
    beta_flux,
    max_flux,
    *,
    n_iter,
    update_fluxes=False,
    collect=False,
):
    lab = kernels.label_f(data, threshold)
    e_fn, m_fn = _uncensored_fns(data, lab, background, alpha, beta)
    flux = (alpha_flux, beta_flux, max_flux) if update_fluxes else None
    return _advance(state, e_fn, m_fn, flux, n_iter, collect)


@jax.jit
def run_mle(data, threshold, times, background):
    lab = kernels.label_f(data, threshold)
    return kernels.mle_fuse(data, lab, times, background)


# Jitted wrappers for the single-image class API. The launcher calls the `_impl`
# functions directly: nesting these jits inside its vmap breaks XLA's sharding
# propagation across the scan mesh.
init_bayesian = partial(jax.jit, static_argnames=("n_init",))(init_bayesian_impl)
advance_bayesian = partial(
    jax.jit, static_argnames=("n_iter", "update_fluxes", "collect")
)(advance_bayesian_impl)
advance_uncensored = partial(
    jax.jit, static_argnames=("n_iter", "update_fluxes", "collect")
)(advance_uncensored_impl)
