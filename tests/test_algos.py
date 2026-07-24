import numpy as np
import pytest

from bayes_mef.algos import (
    BayesianMEF,
    CensoringPrior,
    ConventionalMEF,
    PtychogramMean,
    UncensoredEM,
)

from . import _reference_numpy as ref

SCENARIOS = ["camera_scenario", "diffraction_scenario"]


@pytest.fixture(params=SCENARIOS)
def scenario(request, x64):
    if request.param == "diffraction_scenario" and not x64:
        pytest.skip(
            "16-bit threshold needs double precision; see test_censored_poisson"
        )
    return request.getfixturevalue(request.param)


def test_conventional_mle(scenario, assert_close):
    got = ConventionalMEF(**scenario)
    got.mle()
    want = ref.ConventionalMEF(**scenario)
    want.mle()
    assert_close(got.fused_image, want.fused_image)


@pytest.mark.parametrize("n_iter", [1, 5, 50])
def test_bayesian_matches_reference(scenario, assert_close, n_iter):
    got = BayesianMEF(**scenario)
    got.run(n_iter)
    want = ref.BayesianMEF(**scenario)
    want.run(n_iter)
    assert_close(got.fused_image, want.fused_image)


@pytest.mark.parametrize("n_iter", [1, 20])
def test_uncensored_matches_reference(scenario, assert_close, n_iter):
    got = UncensoredEM(**scenario)
    got.run(n_iter)
    want = ref.UncensoredEM(**scenario)
    want.run(n_iter)
    assert_close(got.fused_image, want.fused_image)


@pytest.mark.parametrize("max_flux", [3, 100])
def test_flux_update_and_gate(scenario, assert_close, max_flux):
    """The flux gate stops refreshing after `max_flux` iterations."""
    kw = dict(scenario, update_fluxes=True, n_iter_fluxes=max_flux)
    got = BayesianMEF(**kw)
    got.run(20)
    want = ref.BayesianMEF(**kw)
    want.run(20)
    assert_close(got.fused_image, want.fused_image)
    assert_close(got.times, want.times)


@pytest.mark.parametrize("prior_cls", [CensoringPrior, PtychogramMean])
def test_prior_assigned_after_construction(scenario, assert_close, prior_cls):
    """The launcher sets alpha/beta post-construction, so the warm start must
    still have used the default prior."""
    ref_prior_cls = getattr(ref, prior_cls.__name__)

    got = BayesianMEF(**scenario)
    p = prior_cls(scenario["data"], scenario["threshold"])
    p()
    got.alpha, got.beta = p.alpha, p.beta
    got.run(20)

    want = ref.BayesianMEF(**scenario)
    rp = ref_prior_cls(scenario["data"], scenario["threshold"])
    rp()
    want.alpha, want.beta = rp.alpha, rp.beta
    want.run(20)

    assert_close(got.fused_image, want.fused_image)


def test_run_is_resumable(scenario, assert_close):
    """Two runs of 10 must equal one run of 20."""
    a = BayesianMEF(**scenario)
    a.run(10)
    a.run(10)
    b = BayesianMEF(**scenario)
    b.run(20)
    assert_close(a.fused_image, b.fused_image)


def test_next_advances_one_iteration(scenario, assert_close):
    a = BayesianMEF(**scenario)
    next(a)
    b = BayesianMEF(**scenario)
    b.run(1)
    assert_close(a.fused_image, b.fused_image)


def test_return_params_shape(scenario, assert_close):
    n_iter = 7
    got = BayesianMEF(**scenario)
    fused = got.run(n_iter, return_params=True)
    assert len(fused) == n_iter + 1
    assert all(isinstance(f, np.ndarray) for f in fused)
    assert_close(fused[-1], got.fused_image)

    with_flux = BayesianMEF(**dict(scenario, update_fluxes=True))
    fused, times = with_flux.run(n_iter, return_params=True)
    assert len(fused) == n_iter + 1
    assert len(times) == n_iter + 1


def test_prior_values_match_reference(scenario, assert_close):
    for cls in (CensoringPrior, PtychogramMean):
        got = cls(scenario["data"], scenario["threshold"])
        got()
        want = getattr(ref, cls.__name__)(scenario["data"], scenario["threshold"])
        want()
        assert_close(got.alpha, want.alpha)
        assert_close(got.beta, want.beta)
