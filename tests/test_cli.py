import jax
import pytest

from bayes_mef.cli import main

_GPU_PLATFORMS = ("gpu", "cuda", "rocm")


def test_check_gpu_matches_environment(capsys):
    """Exit 0 iff a GPU is actually present; the run must not raise."""
    has_gpu = any(d.platform in _GPU_PLATFORMS for d in jax.devices())
    rc = main(["check", "gpu"])
    assert rc == (0 if has_gpu else 1)

    out = capsys.readouterr().out
    assert "JAX" in out
    assert ("working" in out) == has_gpu


@pytest.mark.parametrize("argv", [[], ["check"], ["check", "nonsense"], ["bogus"]])
def test_argparse_errors_exit_2(argv):
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2
