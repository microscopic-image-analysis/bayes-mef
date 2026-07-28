"""Render the MEF regime study as PNG figures under benchmarks/figures/.

Reuses the simulation helpers from ``mef_regime_study.py`` so the plots and the
tables in ``FINDINGS.md`` come from the same code path.

    uv run python benchmarks/plot_findings.py

Colours: EM (blue #0072B2) vs MLE (vermillion #D55E00) — a colourblind-safe
pair (validated, CVD ΔE ~22). Lower error is always better; that direction is
stated on every axis so a reader knows which way is good at a glance.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mef_regime_study import (
    HEAVY,
    LADDER,
    N_SEEDS,
    THRESHOLD,
    _fuse,
    diffraction,
    ratios,
    relerr,
    simulate,
)

from bayes_mef import BayesianMEF

EM_C = "#0072B2"  # BayesianMEF
MLE_C = "#D55E00"  # ConventionalMEF
REF_C = "#4d4d4d"  # oracle / reference (neutral, never a series)
GRID = "#d9d9d9"
INK = "#222222"
MUTED = "#666666"

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 120,
        "font.size": 11,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _style(ax):
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------- #
# Figure 1: where EM beats MLE — error by pixel regime (horizontal bars).
# --------------------------------------------------------------------------- #
def fig_regimes():
    peak, bg = 6, 60
    names = [
        "saturated\nin all exposures",
        "background-\ndominated tail",
        "well-measured",
        "whole frame",
    ]
    masks_of = lambda truth, data: [
        (data >= THRESHOLD).all(axis=0),
        (HEAVY[-1] * truth < bg) & ~(data >= THRESHOLD).all(axis=0),
        ~((data >= THRESHOLD).all(axis=0)) & ~(HEAVY[-1] * truth < bg),
        np.ones(truth.shape, bool),
    ]
    em_v = [[] for _ in names]
    mle_v = [[] for _ in names]
    for s in range(N_SEEDS):
        truth = peak * diffraction(s)
        data = simulate(truth, HEAVY, bg, s)
        em, mle = _fuse(data, HEAVY, bg)
        for i, m in enumerate(masks_of(truth, data)):
            em_v[i].append(relerr(em.fused_image, truth, m))
            mle_v[i].append(relerr(mle.fused_image, truth, m))
    em_m = [np.mean(v) for v in em_v]
    mle_m = [np.mean(v) for v in mle_v]

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    y = np.arange(len(names))[::-1]
    h = 0.36
    ax.barh(y + h / 2, em_m, height=h, color=EM_C, label="BayesianMEF (EM)")
    ax.barh(y - h / 2, mle_m, height=h, color=MLE_C, label="ConventionalMEF (MLE)")
    for yi, ev, mv in zip(y, em_m, mle_m, strict=True):
        ax.text(
            ev + 0.01,
            yi + h / 2,
            f"{ev:.2f}",
            va="center",
            ha="left",
            fontsize=9,
            color=EM_C,
        )
        ax.text(
            mv + 0.01,
            yi - h / 2,
            f"{mv:.2f}",
            va="center",
            ha="left",
            fontsize=9,
            color=MLE_C,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9.5)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("relative error  ←  lower is better")
    ax.set_title(
        "Where BayesianMEF wins: it recovers saturated pixels\n"
        "(MLE discards them → error 1.0); on clean pixels MLE edges ahead",
        fontsize=11.5,
        loc="left",
    )
    _style(ax)
    ax.legend(loc="lower right", frameon=False, fontsize=9.5)
    ax.annotate(
        "MLE recovers\nnothing here",
        xy=(1.0, y[0] - h / 2),
        xytext=(0.62, y[0] - h / 2 - 0.05),
        fontsize=8.5,
        color=MUTED,
        va="center",
        arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 1},
    )
    fig.tight_layout()
    out = os.path.join(FIGDIR, "1_em_vs_mle_by_regime.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------------------- #
# Figure 2: how hard to overexpose — error vs OE, core / tail / whole.
# --------------------------------------------------------------------------- #
def fig_overexpose():
    bg = 30
    OEs = [0.25, 1, 4, 16, 64, 256]
    keys = ("core", "tail", "whole")
    data = {k: {"em": [], "mle": []} for k in keys}
    sat = []
    for OE in OEs:
        times = OE * LADDER
        acc = {k: {"em": [], "mle": []} for k in keys}
        sacc = []
        for s in range(N_SEEDS):
            truth = diffraction(s)
            d = simulate(truth, times, bg, s)
            em, mle = _fuse(d, times, bg)
            regions = {
                "core": truth >= np.quantile(truth, 0.92),
                "tail": truth <= np.quantile(truth, 0.55),
                "whole": np.ones(truth.shape, bool),
            }
            sacc.append((d >= THRESHOLD).mean())
            for k, m in regions.items():
                acc[k]["em"].append(relerr(em.fused_image, truth, m))
                acc[k]["mle"].append(relerr(mle.fused_image, truth, m))
        for k in keys:
            data[k]["em"].append(np.mean(acc[k]["em"]))
            data[k]["mle"].append(np.mean(acc[k]["mle"]))
        sat.append(100 * np.mean(sacc))

    titles = {
        "core": "bright core\n(saturates first)",
        "tail": "weak large-angle tail\n(the information)",
        "whole": "whole pattern\n(what phase retrieval needs)",
    }
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), sharey=True)
    for ax, k in zip(axes, keys, strict=True):
        ax.plot(
            OEs,
            data[k]["mle"],
            "o-",
            color=MLE_C,
            lw=2,
            ms=6,
            label="ConventionalMEF (MLE)",
        )
        ax.plot(
            OEs, data[k]["em"], "s-", color=EM_C, lw=2, ms=6, label="BayesianMEF (EM)"
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(titles[k], fontsize=10.5)
        ax.set_xlabel("overexposure factor  →  more exposure")
        _style(ax)
    # mark the whole-pattern sweet spot
    whole = np.array(data["whole"]["em"])
    best = int(np.argmin(whole))
    axes[2].annotate(
        "sweet spot",
        xy=(OEs[best], whole[best]),
        xytext=(OEs[best], whole[best] * 2.4),
        fontsize=9,
        color=REF_C,
        ha="center",
        arrowprops={"arrowstyle": "->", "color": REF_C, "lw": 1},
    )
    axes[0].set_ylabel("relative error  ←  lower is better")
    axes[0].legend(loc="upper right", frameon=False, fontsize=9)
    fig.suptitle(
        "How hard to overexpose?  The tail keeps improving with exposure "
        "(acquisition win, EM≈MLE);\nthe whole pattern has a sweet spot, "
        "and EM pulls ahead once the core saturates",
        fontsize=11.5,
        x=0.02,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = os.path.join(FIGDIR, "2_overexposure_tradeoff.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------------------- #
# Figure 3: flux-estimation strategy — recovered long-exposure ratio.
# --------------------------------------------------------------------------- #
def fig_flux():
    bg = 30
    OEs = [64, 256, 1024]
    strategies = {
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
        "matched init + update  ★ recommended": {
            "times": None,
            "flux_init": "matched",
            "update_fluxes": True,
        },
    }
    # colour: highlight the recommended strategy, others muted; oracle = ref line
    colors = ["#bdbdbd", "#8c8c8c", "#5aa9df", EM_C]
    censs = []
    top = {name: [] for name in strategies}
    for OE in OEs:
        times_true = OE * LADDER
        acc = {name: [] for name in strategies}
        sacc = []
        for s in range(N_SEEDS):
            truth = diffraction(s)
            d = simulate(truth, times_true, bg, s)
            sacc.append((d >= THRESHOLD).mean())
            for name, kw in strategies.items():
                em = BayesianMEF(d, THRESHOLD, background=bg, **kw)
                em.run(n_iter=250)
                acc[name].append(ratios(em.times)[-1])  # long-exposure ratio
        for name in strategies:
            top[name].append(np.mean(acc[name]))
        censs.append(100 * np.mean(sacc))

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    x = np.arange(len(OEs))
    w = 0.19
    for i, (name, _) in enumerate(strategies.items()):
        off = (i - 1.5) * w
        bars = ax.bar(x + off, top[name], width=w, color=colors[i], label=name)
        for b, v in zip(bars, top[name], strict=True):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + 1.0,
                f"{v:.0f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=INK,
            )
    ax.axhline(64, color=REF_C, lw=1.4, ls="--")
    ax.text(
        x[0] - 0.55,
        64,
        "true ratio = 64 (oracle)",
        va="bottom",
        ha="left",
        fontsize=8.5,
        color=REF_C,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c:.0f}% censored" for c in censs])
    ax.set_ylim(0, 72)
    ax.set_xlim(-0.6, len(OEs) - 0.4)
    ax.set_ylabel("recovered long-exposure flux ratio\n(true = 64, taller is better)")
    ax.set_title(
        "Estimating unknown fluxes: the default collapses under censoring;\n"
        "'matched init + update' tracks the oracle even past 60% censored",
        fontsize=11.5,
        loc="left",
    )
    _style(ax)
    # legend below the plot, out of the bars' way
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.11),
        frameon=False,
        fontsize=9,
        ncol=2,
        columnspacing=2.0,
        handletextpad=0.6,
    )
    fig.tight_layout()
    out = os.path.join(FIGDIR, "3_flux_strategy.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    fig_regimes()
    fig_overexpose()
    fig_flux()
    print("done ->", FIGDIR)
