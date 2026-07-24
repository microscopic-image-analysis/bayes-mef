import numpy as np
import pytest
from scipy.stats import poisson

from bayes_mef._compat import X32_SAFE_THRESHOLD
from bayes_mef.kernels import censored_poisson_mean

from ._reference_numpy import BayesianMEF as RefBayesianMEF

# Integer thresholds and, critically, non-integer ones: `set_threshold` returns
# `max - sqrt(max)`, and SciPy floors before evaluating the survival function.
THRESHOLDS = [1500.0, 2048.0, 1234.7, 1499.5, 65279.002]

# Rate as a fraction of the threshold. Below ~0.7 the survival terms underflow in
# single precision and the expectation degrades to `rate` (see the band test
# below), so float32 is only compared where the algorithm actually operates.
SAFE_FRACTIONS = [0.9, 1.0, 1.1, 2.0, 50.0]
UNDERFLOW_FRACTIONS = [1e-3, 0.5]


@pytest.mark.parametrize("threshold", THRESHOLDS)
def test_matches_scipy_reference(threshold, rtol, x64):
    if not x64 and threshold > X32_SAFE_THRESHOLD:
        pytest.skip("float32 igamma is not accurate enough at this threshold")

    fractions = SAFE_FRACTIONS + (UNDERFLOW_FRACTIONS if x64 else [])
    rate = np.array([f * threshold for f in fractions])

    got = np.asarray(censored_poisson_mean(threshold, rate))
    expected = RefBayesianMEF.censored_poison(threshold, rate)

    assert not np.isnan(got).any()
    np.testing.assert_allclose(got, expected, rtol=rtol)


@pytest.mark.parametrize("threshold", [1234.7, 1499.5, 65279.002])
def test_floor_is_required_for_non_integer_thresholds(threshold, x64):
    """SciPy floors before evaluating sf, so `gammainc(t + 1, r)` is not enough."""
    if not x64:
        pytest.skip("identity is asserted at double precision")
    from jax.scipy.special import gammainc

    rate = np.array([0.9 * threshold, threshold, 2.0 * threshold])
    reference = poisson(rate).sf(threshold)

    floored = np.asarray(gammainc(np.floor(threshold) + 1.0, rate))
    unfloored = np.asarray(gammainc(threshold + 1.0, rate))

    np.testing.assert_allclose(floored, reference, rtol=1e-9)
    assert not np.allclose(unfloored, reference, rtol=1e-9)


def test_both_underflow_falls_back_to_rate():
    """Deep in the tail both terms vanish and the expectation degrades to `rate`."""
    rate = np.array([1e-4, 1.0, 5.0])
    got = np.asarray(censored_poisson_mean(1500.0, rate))
    np.testing.assert_allclose(got, rate, rtol=1e-6)


def test_float32_underflow_band_is_characterised(x64):
    """Pin the known precision hazard rather than pretend it is absent.

    Around `rate ~ 0.3-0.7 * threshold` the survival terms are representable in
    float64 but flush to zero in float32, so single precision takes the
    fallback branch and returns `rate` where the true expectation is ~threshold.
    In practice the uncensored warm-start moves rates past this band before the
    censored kernel is first evaluated, which is why float32 remains the default.
    """
    threshold = 1500.0
    rate = np.array([0.5 * threshold])
    got = float(np.asarray(censored_poisson_mean(threshold, rate))[0])

    if x64:
        assert got == pytest.approx(threshold + 1.5, abs=5.0)
    else:
        assert got == pytest.approx(rate[0])


def test_broadcasts_scalar_threshold_over_array_rate():
    rate = np.linspace(1400.0, 3000.0, 37).reshape(1, 37)
    got = np.asarray(censored_poisson_mean(1500.0, rate))
    assert got.shape == rate.shape
    assert not np.isnan(got).any()
