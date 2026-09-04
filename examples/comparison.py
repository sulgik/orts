"""OR-TS against Beta-TS and Full-TS when the common level moves between batches.

The three policies make different bets about what stays fixed (paper,
Section 3).  Under a common shift only OR-TS's bet holds.  Run:

    python examples/comparison.py
"""
import numpy as np
from orts import LogisticBandit, TSPar

rng = np.random.default_rng(2026)
K, N, T = 5, 100_000, 30
contrasts = np.array([0.0, 0.05, 0.10, 0.15, 0.20])        # arm 4 is best, on the logit scale
arms = [f"arm{i}" for i in range(K)]


def batch(level, allocation):
    out = {}
    for i, a in enumerate(arms):
        n = int(N * allocation[a])
        p = 1 / (1 + np.exp(-(level + contrasts[i])))
        out[a] = [n, int(rng.binomial(n, p))] if n > 0 else [0, 0]
    return out


policies = {"Beta-TS": TSPar(), "Full-TS": LogisticBandit(), "OR-TS": LogisticBandit()}
kwargs = {"Beta-TS": {}, "Full-TS": {"odds_ratios_only": False}, "OR-TS": {}}
alloc = {name: {a: 1 / K for a in arms} for name in policies}
best_share = {name: [] for name in policies}

for t in range(T):
    level = np.log(0.03 / 0.97) + rng.normal(0, 0.3)          # common shock, redrawn every batch
    for name, pol in policies.items():
        obs = batch(level, alloc[name])
        pol.update(obs, **kwargs[name])
        alloc[name] = pol.win_prop(draw=20000, rng=rng)
        best_share[name].append(alloc[name]["arm4"])

for name in policies:
    s = best_share[name]
    print(f"{name:8s} share of traffic on the best arm: after 10 batches {s[9]:.2f}, after 30 {s[-1]:.2f}")
