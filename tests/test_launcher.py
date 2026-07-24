import numpy as np
import pytest

from bayes_mef.algos import BayesianMEF, CensoringPrior, PtychogramMean
from bayes_mef.mef_launcher import LaunchMEF, set_fluxes, set_threshold

from . import _reference_numpy as ref

N_ITER = 12


@pytest.fixture
def stack(scan_stack, x64):
    if not x64:
        pytest.skip("16-bit threshold needs double precision")
    return scan_stack


def _reference_em(stack, prior=None, update_fluxes=False, n_iter=N_ITER):
    """Loop the frozen implementation over scans, as parallel_fusion used to."""
    data = np.moveaxis(stack["pty_stack"], 1, 0)  # (n_scans, E, X, Y)
    fused, fluxes = [], []
    for scan in data:
        em = ref.BayesianMEF(
            scan,
            stack["threshold"],
            stack["times"],
            stack["background"],
            update_fluxes,
            n_iter,
        )
        if prior is not None:
            p = getattr(ref, prior.__name__)(scan, stack["threshold"])
            p()
            em.alpha, em.beta = p.alpha, p.beta
        em.run(n_iter)
        fused.append(em.fused_image.copy())
        fluxes.append(np.asarray(em.times).copy())
    return np.array(fused), np.array(fluxes)


def _launch(stack, **kw):
    return LaunchMEF(
        stack["pty_stack"],
        stack["background"],
        stack["times"],
        stack["threshold"],
        **kw,
    )


def test_run_em_matches_per_scan_reference(stack, assert_close):
    got, _ = _launch(stack).run_em(N_ITER)
    want, _ = _reference_em(stack)
    assert got.shape == want.shape
    assert_close(got, want)


def test_run_mle_matches_per_scan_reference(stack, assert_close):
    got = _launch(stack).run_mle()

    data = np.moveaxis(stack["pty_stack"], 1, 0)
    want = []
    for scan in data:
        mle = ref.ConventionalMEF(
            scan, stack["threshold"], stack["times"], stack["background"]
        )
        mle.mle()
        want.append(mle.fused_image.copy())
    assert_close(got, np.array(want))


@pytest.mark.parametrize("batch_size", [1, 2, 3, 5])
def test_batch_size_does_not_change_results(stack, assert_close, batch_size):
    baseline, _ = _launch(stack).run_em(N_ITER)
    got, _ = _launch(stack).run_em(N_ITER, batch_size=batch_size)
    assert_close(got, baseline)


@pytest.mark.parametrize("prior", [CensoringPrior, PtychogramMean])
def test_priors_match_per_scan_reference(stack, assert_close, prior):
    got, _ = _launch(stack, prior=prior).run_em(N_ITER)
    want, _ = _reference_em(stack, prior=prior)
    assert_close(got, want)


def test_update_fluxes_matches_single_scan(stack, assert_close):
    """Validate the launcher against the single scan class rather than NumPy.

    Flux re-estimation couples every pixel through a shared scale factor, so the
    iteration amplifies rounding: 1e-16 differences in the first step reach ~1e-4
    by iteration 12 on this deliberately noisy fixture, and more on CPU. The
    class itself is checked against the reference in test_algos.py; what the
    launcher adds is the vmap/pmap/chunking machinery, and that must be exact.
    """
    got, got_flux = _launch(stack, update_fluxes=True).run_em(N_ITER)

    data = np.moveaxis(stack["pty_stack"], 1, 0)
    want_flux = []
    for scan in data:
        em = BayesianMEF(
            scan,
            stack["threshold"],
            stack["times"],
            stack["background"],
            True,
            N_ITER,
        )
        em.run(N_ITER)
        want_flux.append(np.asarray(em.times))
    assert_close(got_flux, np.array(want_flux))

    # The launcher then re-runs with the mean of the corrected fluxes.
    corrected = np.mean(want_flux, axis=0)
    want = []
    for scan in data:
        em = BayesianMEF(
            scan, stack["threshold"], corrected, stack["background"], False, N_ITER
        )
        em.run(N_ITER)
        want.append(em.fused_image)
    assert_close(got, np.array(want))


def test_shared_3d_and_per_scan_4d_background_agree(stack, assert_close):
    n_scans = stack["pty_stack"].shape[1]
    bg4 = np.broadcast_to(
        stack["background"][:, None],
        (stack["background"].shape[0], n_scans) + stack["background"].shape[1:],
    ).copy()

    shared, _ = _launch(stack).run_em(N_ITER)
    per_scan, _ = LaunchMEF(
        stack["pty_stack"], bg4, stack["times"], stack["threshold"]
    ).run_em(N_ITER)
    assert_close(per_scan, shared)


def test_ncpus_beyond_device_count_warns(stack):
    with pytest.warns(UserWarning, match="set_cpu_devices"):
        _launch(stack).run_em(1, ncpus=9999)


def test_outputs_are_numpy(stack):
    fused, fluxes = _launch(stack).run_em(1)
    assert isinstance(fused, np.ndarray)
    assert isinstance(fluxes, np.ndarray)
    assert fused.flat is not None  # numpy-only idiom used by utils.corr


def test_set_threshold_and_fluxes_return_host_types(scan_stack):
    threshold = set_threshold(scan_stack["pty_stack"])
    fluxes = set_fluxes(scan_stack["pty_stack"], threshold)
    assert np.isscalar(threshold) or threshold.ndim == 0
    assert isinstance(fluxes, list)
    assert len(fluxes) == scan_stack["pty_stack"].shape[0]
