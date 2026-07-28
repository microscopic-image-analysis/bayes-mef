import warnings

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


def test_threshold_and_times_are_estimated_when_omitted(scenario, assert_close):
    """Omitting them must match passing the same estimates explicitly."""
    from bayes_mef.utils import set_fluxes, set_threshold

    data, background = scenario["data"], scenario["background"]
    threshold = set_threshold(data)
    # the default initialiser is "matched", so the explicit estimate must match it
    times = set_fluxes(data, threshold, background=background, common_mask=True)

    auto = BayesianMEF(data, background=background, update_fluxes=True)
    auto.run(10)
    explicit = BayesianMEF(
        data, threshold, times, background=background, update_fluxes=True
    )
    explicit.run(10)

    assert float(np.asarray(auto.threshold)) == pytest.approx(float(threshold))
    assert_close(auto.fused_image, explicit.fused_image)


def test_conventional_mef_estimates_too(scenario, assert_close):
    """Omitting times uses the default 'matched' initialiser under the hood."""
    from bayes_mef.utils import set_fluxes, set_threshold

    data, background = scenario["data"], scenario["background"]
    threshold = set_threshold(data)
    times = set_fluxes(data, threshold, background=background, common_mask=True)

    auto = ConventionalMEF(data, background=background)
    auto.mle()
    explicit = ConventionalMEF(data, threshold, times, background=background)
    explicit.mle()
    assert_close(auto.fused_image, explicit.fused_image)


def test_estimated_fluxes_warn_unless_refined(scenario):
    """`update_fluxes` defaults to False, so an estimate warns until asked for."""
    data, background = scenario["data"], scenario["background"]

    # Estimated fluxes without refinement warn, whether that is the default
    # or spelled out.
    for kw in ({}, {"update_fluxes": False}):
        with pytest.warns(UserWarning, match="flux factors were estimated"):
            BayesianMEF(data, background=background, **kw)

    # Asking for refinement is what silences it.
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        BayesianMEF(data, background=background, update_fluxes=True)

    # Supplied times are used as given, and say nothing.
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        em = BayesianMEF(data, times=scenario["times"], background=background)
    assert em.update_fluxes is False


def test_estimated_fluxes_plus_refinement_recover_the_truth():
    """Auto flux factors are only a starting point; EM has to fix them up."""
    from skimage.data import camera

    rng = np.random.default_rng(0)
    truth = camera().astype(float)
    times, background, threshold = np.array([0.1, 1.0, 10.0]), 60.0, 1500.0
    data = np.clip(
        [rng.poisson(t * truth + background) for t in times], None, threshold
    ).astype(float)

    def corr(f):
        return float(np.corrcoef(np.asarray(f).ravel(), truth.ravel())[0, 1])

    refined = BayesianMEF(data, background=background, update_fluxes=True)
    refined.run(200)
    assert corr(refined.fused_image) > 0.95

    with pytest.warns(UserWarning):
        raw = BayesianMEF(
            data, background=background, flux_init="uncensored", update_fluxes=False
        )
    raw.run(200)
    assert corr(raw.fused_image) < 0.5  # documents why refinement is on by default


def test_default_flux_estimator_is_unchanged():
    """The default is the v0.1.9 estimator; published results depend on it."""
    from bayes_mef.utils import set_fluxes

    rng = np.random.default_rng(0)
    for shape in [(3, 32, 32), (6, 5, 24, 24)]:
        stack = np.clip(rng.poisson(400, shape), None, 500).astype(float)
        threshold = 500.0

        for use_uncensored in (True, False):
            pty = stack * (stack < threshold) if use_uncensored else stack
            ind = len(pty) // 2 - 1
            axis_last = tuple(range(1, pty.ndim))
            expected = (np.sum(pty, axis=axis_last) + 1e-5) / (np.sum(pty[ind]) + 1e-5)
            got = set_fluxes(stack, threshold, use_uncensored)
            np.testing.assert_array_equal(np.asarray(got), expected)


def test_matched_initialiser_recovers_true_flux_ratios():
    """Background subtraction plus a common mask fixes both sources of bias."""
    from bayes_mef.utils import set_fluxes

    rng = np.random.default_rng(0)
    n = 48
    y, x = np.indices((n, n)) - n / 2
    truth = 1e6 * np.exp(-(x**2 + y**2) / 8.0) + 5e3 / (1.0 + x**2 + y**2)

    times = np.array([2.0**b for b in range(7, 13)])
    background, threshold = 10.0, 65279.0
    data = np.clip(
        [rng.poisson(t * truth / times.max()) + background for t in times],
        None,
        threshold,
    ).astype(float)

    ind = len(times) // 2 - 1
    expected = times / times[ind]

    matched = np.asarray(
        set_fluxes(data, threshold, background=background, common_mask=True)
    )
    np.testing.assert_allclose(matched, expected, rtol=0.05)

    # the default is the one that needs refining
    default = np.asarray(set_fluxes(data, threshold))
    assert np.max(np.abs(default - expected) / expected) > 0.5


def test_matched_initialiser_needs_no_refinement_when_scattering_is_strong():
    """`matched` matches knowing the exposures -- but only at high SNR.

    See test_weakly_scattering_data_needs_the_real_exposures for the regime
    where this does not hold.
    """
    from skimage.data import camera

    rng = np.random.default_rng(0)
    truth = camera().astype(float)
    times, background, threshold = np.array([0.1, 1.0, 10.0]), 60.0, 1500.0
    data = np.clip(
        [rng.poisson(t * truth + background) for t in times], None, threshold
    ).astype(float)

    def corr(f):
        return float(np.corrcoef(np.asarray(f).ravel(), truth.ravel())[0, 1])

    em = BayesianMEF(
        data, background=background, flux_init="matched", update_fluxes=False
    )
    em.run(50)
    assert corr(em.fused_image) > 0.98


def test_estimated_fluxes_warn_for_either_initialiser(scenario):
    """`matched` reduces the compression but does not remove it."""
    for init in ("uncensored", "matched"):
        with pytest.warns(UserWarning):
            BayesianMEF(
                scenario["data"],
                background=scenario["background"],
                flux_init=init,
                update_fluxes=False,
            )


def test_refined_fluxes_are_silent_when_the_estimate_is_informative(scenario):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        BayesianMEF(
            scenario["data"],
            background=scenario["background"],
            flux_init="matched",
            update_fluxes=True,
        )


def test_unknown_flux_init_is_rejected(scenario):
    with pytest.raises(ValueError, match="flux_init must be one of"):
        BayesianMEF(scenario["data"], flux_init="nonsense")


def test_weakly_scattering_data_defeats_every_flux_estimator():
    """Once the background dominates, summed counts carry no flux information.

    This is the low SNR regime the method targets, and the reason estimated
    flux factors are only ever a starting point: what recovers the range is
    EM's iterative treatment of the background, not the initialiser.
    """
    rng = np.random.default_rng(0)
    n = 96
    times = np.array([1.0, 4.0, 16.0, 64.0])
    ind = len(times) // 2 - 1

    y, x = np.indices((n, n)) - n / 2
    peak, background = 8.0, 10.0  # weak scatterer, background ~ signal
    truth = peak * np.exp(-(x**2 + y**2) / 150.0) + 0.05 * peak
    data = np.array([rng.poisson(t * truth / times.max() + background) for t in times])
    threshold = float(data.max() * 0.6)
    data = np.clip(data, None, threshold).astype(float)

    def ratios(em):
        t = np.asarray(em.times)
        return t / t[ind]

    # Neither initialiser resolves the exposures: both come out nearly flat.
    for init in ("uncensored", "matched"):
        with pytest.warns(UserWarning):
            em = BayesianMEF(
                data, background=background, flux_init=init, update_fluxes=False
            )
        assert np.max(ratios(em)) < 2.0, f"{init} unexpectedly resolved the exposures"

    # Refinement recovers most of the range; the true span is 64x.
    refined = BayesianMEF(
        data, background=background, flux_init="matched", update_fluxes=True
    )
    refined.run(300)
    assert np.max(ratios(refined)) > 8.0


def test_degenerate_flux_estimate_warns_even_with_refinement():
    """A nearly flat estimate is a data-quality signal, not just a nudge."""
    rng = np.random.default_rng(0)
    # background swamps the signal entirely
    data = rng.poisson(200.0, (4, 32, 32)).astype(float)

    with pytest.warns(UserWarning, match="carry almost no information"):
        BayesianMEF(data, background=200.0, update_fluxes=True)


def test_per_exposure_background_is_not_spread_over_columns(scenario, assert_close):
    """A `(n_exposures,)` background is one offset per exposure, not per column.

    NumPy lines shapes up from the trailing axis, so before `align_background`
    this silently subtracted a different offset from each image column whenever
    the exposure count happened to match the image width.
    """
    data, threshold = scenario["data"], scenario["threshold"]
    n_exposures = data.shape[0]
    offsets = np.linspace(5.0, 20.0, n_exposures)

    got = BayesianMEF(data, threshold, scenario["times"], offsets)
    got.run(10)
    want = BayesianMEF(
        data, threshold, scenario["times"], offsets.reshape(n_exposures, 1, 1)
    )
    want.run(10)
    assert_close(got.fused_image, want.fused_image)


def test_background_shape_mismatch_is_rejected(scenario):
    data = scenario["data"]
    with pytest.raises(ValueError, match="does not fit a stack of shape"):
        BayesianMEF(data, background=np.zeros(data.shape[0] + 1))
