"""Compare the frozen NumPy+joblib implementation against the JAX port.

Usage (each configuration needs its own process, since device count and
precision are fixed at JAX startup):

    uv run python benchmarks/bench_mef.py numpy
    JAX_PLATFORMS=cpu uv run python benchmarks/bench_mef.py jax
    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=16 \
        uv run python benchmarks/bench_mef.py jax
    uv run python benchmarks/bench_mef.py jax            # GPU
"""

import os
import sys
import time

import numpy as np

N_SCANS = int(os.environ.get("BENCH_SCANS", "64"))
N_ITER = int(os.environ.get("BENCH_ITERS", "150"))
SIZE = int(os.environ.get("BENCH_SIZE", "256"))


def make_stack():
    rng = np.random.default_rng(0)
    y, x = np.indices((SIZE, SIZE)) - SIZE / 2
    r2 = x**2 + y**2
    truth = 1e6 * np.exp(-r2 / 8.0) + 5e3 / (1.0 + r2)

    times = np.array([2.0**b for b in range(7, 13)], dtype=float)
    threshold = float(65535.0 - np.sqrt(65535.0))
    base = np.stack([rng.poisson(t * truth / times.max()) for t in times])

    # (n_exposures, n_scans, X, Y)
    stack = np.stack(
        [np.clip(rng.poisson(base), None, threshold) for _ in range(N_SCANS)], axis=1
    ).astype(float)
    background = np.full(stack.shape[0:1] + stack.shape[2:], 10.0)
    return stack, background, times, threshold


def run_numpy(stack, background, times, threshold, ncpus):
    from joblib import Parallel, delayed

    sys.path.insert(0, "tests")
    from _reference_numpy import BayesianMEF

    data = np.moveaxis(stack, 1, 0)

    def one(scan):
        em = BayesianMEF(scan, threshold, times, background)
        em.run(N_ITER)
        return em.fused_image.copy()

    with Parallel(n_jobs=ncpus) as parallel:
        fused = parallel(delayed(one)(scan) for scan in data)
    return np.array(fused)


def run_jax(stack, background, times, threshold, ncpus):
    from bayes_mef.mef_launcher import LaunchMEF

    launch = LaunchMEF(stack, background, times, threshold)
    fused, _ = launch.run_em(N_ITER, ncpus)
    return fused


if __name__ == "__main__":
    backend = sys.argv[1] if len(sys.argv) > 1 else "jax"
    ncpus = int(sys.argv[2]) if len(sys.argv) > 2 else None

    stack, background, times, threshold = make_stack()
    print(f"{N_SCANS} scans x {stack.shape[0]} exposures x {SIZE}^2, {N_ITER} iters")

    if backend == "numpy":
        label = f"numpy + joblib (n_jobs={ncpus})"
        start = time.perf_counter()
        fused = run_numpy(stack, background, times, threshold, ncpus or 1)
        elapsed = time.perf_counter() - start
    else:
        import jax

        label = f"jax [{jax.default_backend()}, {jax.device_count()} dev, "
        label += "f64]" if jax.config.jax_enable_x64 else "f32]"
        # Time a second run so compilation is not counted.
        run_jax(stack[:, :2], background, times, threshold, ncpus)
        start = time.perf_counter()
        fused = run_jax(stack, background, times, threshold, ncpus)
        elapsed = time.perf_counter() - start

    print(f"  {label:42s} {elapsed:8.2f} s   checksum={float(fused.sum()):.6e}")
