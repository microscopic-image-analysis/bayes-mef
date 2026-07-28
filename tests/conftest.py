import os

import numpy as np
import pytest

X64 = os.environ.get("JAX_ENABLE_X64", "0") not in ("0", "", "false", "False")


@pytest.fixture(scope="session")
def x64():
    return X64


@pytest.fixture(scope="session")
def rtol():
    """Tolerance for differential comparisons against the NumPy reference.

    Under x64 the port should agree to near machine precision; the 1e-9 floor
    leaves room for XLA's igamma, which drifts to ~2e-10 at a = 65535.
    """
    return 1e-9 if X64 else 2e-3


# Fraction of elements allowed to exceed rtol.
#
# `censored_poisson_mean` falls back to `rate` when both survival terms underflow
# to exactly zero. That branch is a discrete function of where the incomplete
# gamma implementation underflows, so SciPy, XLA-CPU and XLA-GPU disagree on a
# thin band of inputs -- a property of the algorithm, not of the port.
#
# In double precision only pixels sitting exactly on that boundary flip (~1 in
# 11520 measured on the scan fixture). In single precision the band is far wider,
# because float32 underflows at 1e-38 rather than 1e-308 (~0.007% of pixels on
# the camera scenario). Everything off the band agrees to `rtol`.
OUTLIER_BUDGET = 1e-3


@pytest.fixture(scope="session")
def assert_close(rtol):
    """Compare against the reference, tolerating fallback-boundary flips."""

    def _assert(got, want, err_msg=""):
        got, want = np.asarray(got), np.asarray(want)
        bad = ~np.isclose(got, want, rtol=rtol, atol=0)
        fraction = bad.mean()
        assert fraction <= OUTLIER_BUDGET, (
            f"{fraction:.4%} of elements exceed rtol={rtol} "
            f"(budget {OUTLIER_BUDGET:.2%}). {err_msg}"
        )

    return _assert


@pytest.fixture(scope="session")
def camera_scenario():
    """The README/demo scenario: 3 exposures of the camera image, integer threshold."""
    from skimage.data import camera

    rng = np.random.default_rng(0)
    truth = camera().astype(float)
    times = np.array([0.1, 1.0, 10.0])
    background = 60.0
    threshold = 1500.0
    data = np.stack([rng.poisson(t * truth + background) for t in times])
    return {
        "data": np.clip(data, None, threshold).astype(float),
        "threshold": threshold,
        "times": times,
        "background": background,
    }


@pytest.fixture(scope="session")
def diffraction_scenario():
    """Ptychography-like: wide dynamic range, many exposures, non-integer threshold."""
    rng = np.random.default_rng(1)
    n = 48
    y, x = np.indices((n, n)) - n / 2
    r2 = x**2 + y**2
    truth = 1e6 * np.exp(-r2 / 8.0) + 5e3 / (1.0 + r2)

    times = np.array([2.0**b for b in range(7, 13)], dtype=float)
    background = 10.0
    data = np.stack([rng.poisson(t * truth / times.max()) for t in times])

    limit = 65535.0
    threshold = float(limit - np.sqrt(limit))  # non-integer, as set_threshold produces
    return {
        "data": np.clip(data, None, threshold).astype(float),
        "threshold": threshold,
        "times": times,
        "background": background,
    }


@pytest.fixture(scope="session")
def scan_stack(diffraction_scenario):
    """A small multi-scan stack shaped (n_exposures, n_scans, X, Y) for LaunchMEF."""
    rng = np.random.default_rng(2)
    base = diffraction_scenario
    n_scans = 5
    stack = np.stack(
        [
            np.clip(
                rng.poisson(np.clip(base["data"], 0, None)), None, base["threshold"]
            ).astype(float)
            for _ in range(n_scans)
        ],
        axis=1,
    )
    return {
        "pty_stack": stack,  # (n_exposures, n_scans, X, Y)
        "threshold": base["threshold"],
        "times": base["times"],
        "background": np.full(stack.shape[0:1] + stack.shape[2:], 10.0),  # (E, X, Y)
    }
