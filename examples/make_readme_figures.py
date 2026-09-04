"""Regenerate the two figures embedded in README.md into docs/.

    pip install -e ".[examples]"   # matplotlib
    python examples/make_readme_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from orts import LogisticBandit, TSPar

OUT = Path(__file__).resolve().parent.parent / "docs"
OUT.mkdir(exist_ok=True)
COLORS = ["#4878a8", "#e08214", "#33883d"]


def rates_and_contrasts() -> None:
    """One synthetic experiment seen twice: absolute rates, then contrasts."""
    rng = np.random.default_rng(30)
    T, n = 40, 200_000
    beta = np.array([0.0, 0.35, 0.70])
    steps = rng.normal(0.0, 0.16, size=T)
    alpha = np.log(0.01 / 0.99) + np.cumsum(steps) - np.cumsum(steps).mean()
    p = 1 / (1 + np.exp(-(alpha[:, None] + beta[None, :])))
    phat = rng.binomial(n, p) / n
    bhat = np.log(phat / (1 - phat)) - np.log(phat[:, :1] / (1 - phat[:, :1]))
    t = np.arange(1, T + 1)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.1))
    for k, label in enumerate(["arm A (ref)", "arm B", "arm C"]):
        axes[0].plot(t, 100 * phat[:, k], lw=1.2, color=COLORS[k], label=label)
    axes[0].set_ylabel("observed event rate (%)")
    axes[0].set_title("what the dashboard shows", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8)
    for k in (1, 2):
        axes[1].plot(t, bhat[:, k], lw=1.2, color=COLORS[k])
        axes[1].axhline(beta[k], color=COLORS[k], lw=0.9, ls="--", alpha=0.6)
    axes[1].axhline(0.0, color="0.75", lw=0.6)
    axes[1].set_ylabel(r"estimated contrast $\hat\beta_{i,t}$")
    axes[1].set_title("the same batches, in contrast coordinates", fontsize=10)
    axes[1].set_ylim(-0.15, 1.0)
    for ax in axes:
        ax.set_xlabel("batch")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "rates_and_contrasts.png", dpi=160)
    plt.close(fig)


def comparison_under_common_shock() -> None:
    """Share of traffic on the best arm, three policies, a common shock every batch."""
    K, N, T, reps = 5, 100_000, 30, 5
    contrasts = np.array([0.0, 0.05, 0.10, 0.15, 0.20])
    arms = [f"arm{i}" for i in range(K)]
    specs = {"Beta-TS": (TSPar, {}), "Full-TS": (LogisticBandit, {"odds_ratios_only": False}),
             "OR-TS": (LogisticBandit, {})}
    curves = {name: np.zeros((reps, T)) for name in specs}
    for r in range(reps):
        rng = np.random.default_rng(100 + r)
        levels = np.log(0.03 / 0.97) + rng.normal(0, 0.3, size=T)
        for name, (cls, kw) in specs.items():
            pol, alloc = cls(), {a: 1 / K for a in arms}
            for t in range(T):
                obs = {}
                for i, a in enumerate(arms):
                    n = int(N * alloc[a])
                    p = 1 / (1 + np.exp(-(levels[t] + contrasts[i])))
                    obs[a] = [n, int(rng.binomial(n, p))] if n > 0 else [0, 0]
                pol.update(obs, **kw)
                alloc = pol.win_prop(draw=20_000, rng=rng)
                curves[name][r, t] = alloc["arm4"]
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    styles = {"Beta-TS": ("0.35", "--"), "Full-TS": (COLORS[1], ":"), "OR-TS": (COLORS[0], "-")}
    for name, c in curves.items():
        ax.plot(np.arange(1, T + 1), c.mean(axis=0), color=styles[name][0], ls=styles[name][1], lw=1.6, label=name)
    ax.set_xlabel("batch")
    ax.set_ylabel("share of traffic on the best arm")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("common shock (sd 0.30) redrawn every batch, five arms", fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "comparison_common_shock.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    rates_and_contrasts()
    comparison_under_common_shock()
    print("wrote", OUT / "rates_and_contrasts.png", "and", OUT / "comparison_common_shock.png")
