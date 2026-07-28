import warnings

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


def _background_forms(stack):
    """Every way of writing the same constant background this fixture uses."""
    n_exposures, n_scans, *image = stack["pty_stack"].shape
    level = float(stack["background"].flat[0])
    frame = np.full(image, level)
    return {
        "scalar": level,
        "one_element": np.array([level]),
        "per_exposure": np.full(n_exposures, level),
        "single_frame": frame,
        "per_exposure_frames": np.broadcast_to(frame, (n_exposures, *image)).copy(),
        "broadcast_per_image": np.broadcast_to(frame, (n_exposures, n_scans, *image)),
        "per_image": np.full((n_exposures, n_scans, *image), level),
    }


@pytest.mark.parametrize(
    "form",
    [
        "scalar",
        "one_element",
        "per_exposure",
        "single_frame",
        "per_exposure_frames",
        "broadcast_per_image",
        "per_image",
    ],
)
def test_background_forms_agree(stack, assert_close, form):
    """A background means the same thing however the experiment recorded it."""
    baseline, _ = _launch(stack).run_em(N_ITER)
    got, _ = LaunchMEF(
        stack["pty_stack"],
        _background_forms(stack)[form],
        stack["times"],
        stack["threshold"],
    ).run_em(N_ITER)
    assert_close(got, baseline)


def test_broadcast_background_is_not_uploaded_per_scan(stack):
    """`np.broadcast_to(frame, stack.shape)` must not cost a per-scan copy."""
    n_exposures, n_scans, *image = stack["pty_stack"].shape
    frame = np.full(image, float(stack["background"].flat[0]))

    launcher = LaunchMEF(
        stack["pty_stack"],
        np.broadcast_to(frame, (n_exposures, n_scans, *image)),
        stack["times"],
        stack["threshold"],
    )
    assert launcher.background.shape == (n_exposures, *image)


def test_background_shape_mismatch_is_rejected(stack):
    n_exposures = stack["pty_stack"].shape[0]
    with pytest.raises(ValueError, match="does not fit a stack of shape"):
        LaunchMEF(stack["pty_stack"], np.zeros(n_exposures + 1))


def test_estimated_times_warn_unless_refined(stack):
    """`update_fluxes` defaults off here too, so the launcher has to say so."""
    with pytest.warns(UserWarning, match="flux factors were estimated"):
        launcher = LaunchMEF(stack["pty_stack"], stack["background"])
    assert launcher.update_fluxes is False

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        LaunchMEF(stack["pty_stack"], stack["background"], update_fluxes=True)

    # Supplied times are used as given and say nothing.
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        _launch(stack)


def test_ncpus_is_ignored_but_accepted(stack, assert_close):
    """Old call sites pass ncpus positionally; it must not break them."""
    with pytest.warns(DeprecationWarning, match="ncpus=6 is ignored"):
        got, _ = _launch(stack).run_em(N_ITER, 6)

    baseline, _ = _launch(stack).run_em(N_ITER)
    assert_close(got, baseline)


def test_chunking_bounds_device_transfers(stack, monkeypatch):
    """The stack must never be uploaded whole, however small `batch_size` is.

    A ptychogram is easily tens of GB, so transferring it up front exhausts
    device memory before the first chunk runs. That is invisible on CPU, where
    host memory is far larger, which is why this is asserted rather than assumed.
    """
    import jax.numpy as jnp

    from bayes_mef import mef_launcher

    seen = []
    real_asarray = jnp.asarray

    def spy(a, *args, **kwargs):
        out = real_asarray(a, *args, **kwargs)
        if getattr(out, "ndim", 0) >= 3:
            seen.append(int(np.prod(out.shape)))
        return out

    monkeypatch.setattr(mef_launcher.jnp, "asarray", spy)
    _launch(stack).run_em(1, batch_size=1)

    n_exposures, n_scans = stack["pty_stack"].shape[:2]
    per_scan = n_exposures * int(np.prod(stack["pty_stack"].shape[2:]))
    assert seen, "expected at least one device transfer"
    assert max(seen) < per_scan * n_scans, (
        f"largest transfer was {max(seen)} elements but one scan is {per_scan}; "
        "the whole stack was uploaded up front"
    )


def test_batch_size_defaults_to_something_usable(stack):
    from bayes_mef.mef_launcher import _default_batch_size

    n = _default_batch_size(64, (6, 256, 256), 4)
    assert 1 <= n <= 64


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
