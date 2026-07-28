"""Command-line entry point for bayes_mef.

Usage:
    bayes_mef check gpu    # verify a GPU backend is available and working
"""

import argparse
import sys

_GPU_PLATFORMS = ("gpu", "cuda", "rocm")


def _check_gpu() -> int:
    """Verify JAX can see a GPU and actually run a computation on it.

    Returns a process exit code: 0 if a GPU is present and works, 1 otherwise.
    """
    try:
        import jax
        import jax.numpy as jnp
    except Exception as exc:  # noqa: BLE001 - report cleanly if the backend won't import
        print(f"error: could not import JAX ({exc})", file=sys.stderr)
        return 1

    print(f"JAX {jax.__version__}, default backend: {jax.default_backend()}")
    devices = jax.devices()
    for d in devices:
        print(f"  device: {d.platform}:{d.id}  {getattr(d, 'device_kind', '?')}")

    gpus = [d for d in devices if d.platform in _GPU_PLATFORMS]
    if not gpus:
        print("\n✗ No GPU detected — bayes_mef will run on CPU.")
        print(
            '  Install a GPU build, e.g. pip install "bayes_mef[cuda12]" '
            "(or [cuda13]), and make sure the CUDA driver is visible.",
        )
        return 1

    gpu = gpus[0]
    # A real computation, placed on the GPU and read back. Summing 1024 ones in
    # float32 is exact (< 2**24), so every element of a @ a must equal 1024.
    try:
        a = jax.device_put(jnp.ones((1024, 1024), jnp.float32), gpu)
        out = (a @ a).block_until_ready()
        value = float(out[0, 0])
        on_gpu = all(d.platform in _GPU_PLATFORMS for d in out.devices())
    except Exception as exc:  # noqa: BLE001 - any backend error means the GPU is unusable
        print(f"\n✗ GPU found but a test computation failed: {exc}", file=sys.stderr)
        return 1

    if not on_gpu or value != 1024.0:
        print(
            f"\n✗ GPU computation returned an unexpected result "
            f"(value={value}, on_gpu={on_gpu}).",
            file=sys.stderr,
        )
        return 1

    plural = "s" if len(gpus) > 1 else ""
    print(
        f"\n✓ GPU is available and working: {gpu.device_kind} ({len(gpus)} device{plural})."
    )
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="bayes_mef", description="bayes_mef command-line utilities."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="run environment checks")
    check.add_argument(
        "target", choices=["gpu"], help="what to check (currently only 'gpu')"
    )

    parser.parse_args(argv)
    # argparse enforces command == "check" and target == "gpu", the only options.
    return _check_gpu()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
