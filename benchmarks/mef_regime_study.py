"""Where BayesianMEF (EM) beats ConventionalMEF (MLE), and how to acquire/fuse.

A reproducible study behind ``benchmarks/FINDINGS.md``. Every experiment gives
*both* methods the true threshold, exposure times and background, so it compares
the fusion algorithm, not flux/threshold estimation (except the ``flux``
experiment, which is explicitly about flux estimation).

Target is a diffraction pattern ``|FFT(object)|**2`` with the paper's HDR
exposure ladder: a bright core that saturates the long exposures and a weak
large-angle tail near the noise floor. Quality is measured on amplitude/log
scale, which is what phase retrieval consumes; a plain whole-image correlation
saturates near 1.0 and hides every effect of interest.

Usage::

    uv run python benchmarks/mef_regime_study.py regimes     # EM vs MLE, by pixel type
    uv run python benchmarks/mef_regime_study.py overexpose  # how hard to overexpose
    uv run python benchmarks/mef_regime_study.py flux        # flux-estimation strategies
    uv run python benchmarks/mef_regime_study.py all
"""

import sys
import warnings

import numpy as np
from skimage.data import camera
from skimage.transform import resize

from bayes_mef import BayesianMEF, ConventionalMEF

warnings.filterwarnings("ignore")

N = 128
THRESHOLD = 2.0**11  # detector saturation limit
LADDER = np.array([1.0, 8.0, 64.0])  # near-floor / moderate / heavy ratios
HEAVY = np.array([2.0**7, 2.0**10, 2.0**13])  # fixed saturating ladder (the paper's)
N_ITER = 200
N_SEEDS = 4


def diffraction(seed):
    """HDR diffraction-like pattern, normalised so the tail is ~O(1)."""
    r = np.random.default_rng(seed)
    obj = resize(camera().astype(float), (N, N), anti_aliasing=True)
    obj /= obj.max()
    field = obj * np.exp(1j * 2 * np.pi * r.random((N, N)))
    dp = np.abs(np.fft.fftshift(np.fft.fft2(field))) ** 2
    return dp / dp.mean()


def simulate(truth, times, background, seed):
    """Poisson HDR stack, clipped at the censoring threshold."""
    r = np.random.default_rng(1000 + seed)
    data = np.array([r.poisson(t * truth + background).astype(float) for t in times])
    return np.clip(data, None, THRESHOLD)


def relerr(a, b, mask=None):
    """Scale-invariant relative L2 on amplitude (what phase retrieval sees)."""
    a, b = np.sqrt(np.clip(a, 0, None)), np.sqrt(np.clip(b, 0, None))
    s = (a * b).sum() / (a * a).sum()
    a = a * s
    if mask is not None:
        a, b = a[mask], b[mask]
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def ratios(t):
    t = np.asarray(t, float)
    return t / t.min()


def _fuse(data, times, background, **kw):
    mle = ConventionalMEF(data, THRESHOLD, times, background)
    mle.mle()
    em = BayesianMEF(data, THRESHOLD, times, background, **kw)
    em.run(n_iter=N_ITER)
    return em, mle


# --------------------------------------------------------------------------- #
# 1. EM vs MLE, error decomposed by pixel regime + a saturation sweep.
# --------------------------------------------------------------------------- #
def experiment_regimes():
    # Fixed saturating ladder; `peak` scales the target so the core saturates.
    print("\n== EM vs MLE: error by pixel regime (peak=6, bg=60) ==\n")
    peak, bg = 6, 60
    names = [
        "saturated in all exposures",
        "background-dominated tail",
        "well-measured",
        "whole frame",
    ]
    acc = {n: {"em": [], "mle": []} for n in names}
    for s in range(N_SEEDS):
        truth = peak * diffraction(s)
        data = simulate(truth, HEAVY, bg, s)
        em, mle = _fuse(data, HEAVY, bg)
        sat_all = (data >= THRESHOLD).all(axis=0)
        weak = (HEAVY[-1] * truth < bg) & ~sat_all
        well = ~sat_all & ~weak
        masks = [sat_all, weak, well, np.ones_like(sat_all, bool)]
        for n, m in zip(names, masks, strict=True):
            if m.sum() == 0:
                continue
            acc[n]["em"].append(relerr(em.fused_image, truth, m))
            acc[n]["mle"].append(relerr(mle.fused_image, truth, m))
    print(f"{'regime':>28} | {'EM relerr':>10} {'MLE relerr':>10}")
    print("-" * 54)
    for n in names:
        if not acc[n]["em"]:
            continue
        print(
            f"{n:>28} | {np.mean(acc[n]['em']):>10.4f} {np.mean(acc[n]['mle']):>10.4f}"
        )
    print("\n  -> EM's win is concentrated on saturated pixels (MLE discards them,")
    print("     relerr ~1.0); on well-measured pixels MLE is slightly better")
    print("     (unbiased ML vs EM's mild prior bias).")

    print("\n== Saturation sweep (whole-frame relerr, bg=20) ==\n")
    print(f"{'peak':>6} {'sat%':>7} | {'EM':>8} {'MLE':>8} {'MLE/EM':>7}")
    print("-" * 42)
    for peak in [2, 5, 10, 50]:
        em_e, mle_e, sat = [], [], []
        for s in range(N_SEEDS):
            truth = peak * diffraction(s)
            data = simulate(truth, HEAVY, 20, s)
            em, mle = _fuse(data, HEAVY, 20)
            em_e.append(relerr(em.fused_image, truth))
            mle_e.append(relerr(mle.fused_image, truth))
            sat.append((data >= THRESHOLD).mean())
        ee, me = np.mean(em_e), np.mean(mle_e)
        print(
            f"{peak:>6} {100 * np.mean(sat):>6.1f}% | {ee:>8.3f} {me:>8.3f} "
            f"{me / ee:>6.2f}x"
        )


# --------------------------------------------------------------------------- #
# 2. Acquisition strategy: how hard should the long frame overexpose?
# --------------------------------------------------------------------------- #
def experiment_overexpose():
    print("\n== How hard to overexpose? (ladder = OE * [1,8,64], bg=30) ==\n")
    bg = 30
    print(
        f"{'OE':>6} {'cens%':>7} | {'core EM/MLE':>16} | {'tail EM/MLE':>16} | "
        f"{'whole EM/MLE':>16}"
    )
    print("-" * 74)
    for OE in [0.25, 1, 4, 16, 64, 256]:
        times = OE * LADDER
        keys = ("core", "tail", "whole")
        acc = {k: {"em": [], "mle": []} for k in keys}
        sat = []
        for s in range(N_SEEDS):
            truth = diffraction(s)
            data = simulate(truth, times, bg, s)
            em, mle = _fuse(data, times, bg)
            core = truth >= np.quantile(truth, 0.92)
            tail = truth <= np.quantile(truth, 0.55)
            whole = np.ones_like(core, bool)
            sat.append((data >= THRESHOLD).mean())
            for k, m in zip(keys, (core, tail, whole), strict=True):
                acc[k]["em"].append(relerr(em.fused_image, truth, m))
                acc[k]["mle"].append(relerr(mle.fused_image, truth, m))
        cells = {
            k: f"{np.mean(acc[k]['em']):.3f}/{np.mean(acc[k]['mle']):.3f}" for k in keys
        }
        print(
            f"{OE:>6g} {100 * np.mean(sat):>6.1f}% | {cells['core']:>16} | "
            f"{cells['tail']:>16} | {cells['whole']:>16}"
        )
    print("\n  -> The tail (the information) improves monotonically with exposure and")
    print("     is an acquisition effect (EM~=MLE). Whole-frame error has a sweet")
    print("     spot; EM pulls ahead once the core saturates, extending usable OE.")


# --------------------------------------------------------------------------- #
# 3. Flux-estimation strategy under heavy censoring.
# --------------------------------------------------------------------------- #
def experiment_flux():
    print("\n== Flux-estimation strategies (true ratios 1:8:64, bg=30) ==\n")
    bg = 30
    print(f"{'scale':>6} {'cens%':>7} | {'strategy':>26} | {'relerr':>7} | ratios")
    print("-" * 74)
    for OE in [64, 256, 1024]:
        times_true = OE * LADDER
        strategies = {
            "oracle (true times)": {"times": times_true, "update_fluxes": False},
            "uncensored init (default)": {
                "times": None,
                "flux_init": "uncensored",
                "update_fluxes": False,
            },
            "uncensored init + update": {
                "times": None,
                "flux_init": "uncensored",
                "update_fluxes": True,
            },
            "matched init only": {
                "times": None,
                "flux_init": "matched",
                "update_fluxes": False,
            },
            "matched init + update": {
                "times": None,
                "flux_init": "matched",
                "update_fluxes": True,
            },
        }
        rows = {n: {"e": [], "r": []} for n in strategies}
        sat = []
        for s in range(N_SEEDS):
            truth = diffraction(s)
            data = simulate(truth, times_true, bg, s)
            sat.append((data >= THRESHOLD).mean())
            for name, kw in strategies.items():
                em = BayesianMEF(data, THRESHOLD, background=bg, **kw)
                em.run(n_iter=250)
                rows[name]["e"].append(relerr(em.fused_image, truth))
                rows[name]["r"].append(ratios(em.times))
        cens = 100 * np.mean(sat)
        for name in strategies:
            e = np.mean(rows[name]["e"])
            rr = np.mean(rows[name]["r"], axis=0)
            print(
                f"{OE:>6g} {cens:>6.1f}% | {name:>26} | {e:>7.4f} | "
                f"{rr[0]:.2f}:{rr[1]:.2f}:{rr[2]:.2f}"
            )
        print("-" * 74)
    print("  -> matched init + update reaches the oracle and is stable to >60%")
    print("     censoring. The DEFAULT uncensored init collapses (long flux < mid).")


EXPERIMENTS = {
    "regimes": experiment_regimes,
    "overexpose": experiment_overexpose,
    "flux": experiment_flux,
}


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "all":
        for fn in EXPERIMENTS.values():
            fn()
    elif which in EXPERIMENTS:
        EXPERIMENTS[which]()
    else:
        sys.exit(
            f"unknown experiment {which!r}; choose from {list(EXPERIMENTS) + ['all']}"
        )


if __name__ == "__main__":
    main()
