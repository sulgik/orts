"""Contextual OR-TS against GrowthBook's contextual bandit engine, on binomial metrics.

GrowthBook's contextual bandit (`packages/stats-ts/src/contextualBanditWeights.ts`) takes the
decision metric pooled over the whole run per (context, variation), fits a greedy regression tree
over the context attributes, and runs Gaussian Thompson sampling inside each leaf. This script
runs that engine itself, bundled from GrowthBook's source and driven over stdin by
`gb_contextual_driver.ts`, next to `orts.ContextualLogisticBandit`, which fits one logistic model
over arm-by-cell contrasts with a fresh intercept per cell and period.

Environment: K arms and two context attributes, region (3 levels) and device (2 levels), so six
cells of unequal size. Arm a's log-odds contrast in cell (r, d) is

    main[a] + region_effect[a, r] + device_effect[a, d],

the effects drawn once per run with sd `h`; `h = 0` means the best arm is the same everywhere.
The structure is additive in the attributes, which is the case a tree that splits one attribute
at a time is built for. Each cell has its own base rate, and every cell and arm is moved each
period by a common shift (sd `sigma`). Regret is counted against each cell's own best arm.

    python ctxbench.py --driver /path/to/driver.cjs --reps 20
"""
from __future__ import annotations

import argparse
import json
import subprocess
import warnings

import numpy as np
from gbstats.bayesian.bandits import BanditConfig, BanditsSimple
from gbstats.bayesian.tests import GaussianPrior

from gbbench import gb_row
from orts import ContextualLogisticBandit, LogisticBandit

warnings.filterwarnings("ignore")

MAIN = np.array([0.0, 0.05, 0.10, 0.15])  # log-odds; the last arm is best on average
K = len(MAIN)
ARMS = [f"a{i}" for i in range(K)]
REGIONS, REGION_SHARE = ["r0", "r1", "r2"], np.array([0.5, 0.3, 0.2])
DEVICES, DEVICE_SHARE = ["d0", "d1"], np.array([0.7, 0.3])
CELLS = [(r, d) for r in range(len(REGIONS)) for d in range(len(DEVICES))]
C = len(CELLS)
CELL_SHARE = np.array([REGION_SHARE[r] * DEVICE_SHARE[d] for r, d in CELLS])
BASE = np.log(0.03 / 0.97)  # 3% base rate
BASE_SD = 0.3  # spread of the cells' base log-odds
N, T = 20_000, 40  # users per period, periods
FLOOR = 0.01  # GrowthBook's minimum variation weight, used for every policy
MAX_LEAVES = 8  # more than the six cells, so the tree's own BIC rule decides
DRAWS = 20_000

DRIVER: str = ""


def gb_contextual():
    """GrowthBook's contextual engine, fed the cumulative (context, variation) counts."""
    engine = subprocess.Popen(["node", DRIVER], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    n, s = np.zeros((C, K)), np.zeros((C, K))
    metric = {"id": "m", "name": "m", "inverse": False, "statistic_type": "mean",
              "main_metric_type": "binomial"}
    leaves = [1]

    def step(n_t, s_t):
        nonlocal n, s
        n, s = n + n_t, s + s_t
        observations = [
            {"variationIndex": k, "context": {"region": REGIONS[r], "device": DEVICES[d]},
             "arm": {"n": n[c, k], "main_sum": s[c, k], "main_sum_squares": s[c, k],
                     "denominator_sum": 0, "denominator_sum_squares": 0,
                     "main_denominator_sum_product": 0, "covariate_sum": 0,
                     "covariate_sum_squares": 0, "main_covariate_sum_product": 0}}
            for c, (r, d) in enumerate(CELLS) for k in range(K) if n[c, k] > 0]
        engine.stdin.write(json.dumps({
            "varIds": ARMS, "attributes": ["region", "device"], "maxLeaves": MAX_LEAVES,
            "minUsersPerLeaf": 100, "metricSettings": metric,
            "analysisWeights": [1 / K] * K, "observations": observations}) + "\n")
        engine.stdin.flush()
        responses = json.loads(engine.stdout.readline())["responses"]
        weights = np.full((C, K), 1 / K)
        for resp in responses:
            cell = CELLS.index((REGIONS.index(resp["context"]["region"]),
                                DEVICES.index(resp["context"]["device"])))
            weights[cell] = resp["weights"]
        leaves[0] = len({resp["leafId"] for resp in responses})
        return weights

    step.leaves = leaves
    step.close = engine.kill
    return step


def gb_plain():
    """GrowthBook's non-contextual bandit as shipped: period-weighted, context ignored."""
    cfg = BanditConfig(prior_distribution=GaussianPrior(mean=0, variance=1e4, proper=True),
                       weight_by_period=True, top_two=False, min_variation_weight=FLOOR,
                       bandit_weights_seed=100)
    ns, ss, alloc = [], [], np.full(K, 1 / K)

    def step(n_t, s_t):
        nonlocal alloc
        ns.append(n_t.sum(0))
        ss.append(s_t.sum(0))
        stats = gb_row(np.array(ns), np.array(ss), True, True)
        result = BanditsSimple(stats, list(alloc), cfg).compute_result()
        if result.bandit_weights is not None:
            alloc = np.array(result.bandit_weights)
        return np.tile(alloc, (C, 1))

    return step


def orts_pooled():
    """OR-TS with the context ignored."""
    bandit, rng = LogisticBandit(), np.random.default_rng(1)

    def step(n_t, s_t):
        n_k, s_k = n_t.sum(0), s_t.sum(0)
        bandit.update({a: [n_k[i], s_k[i]] for i, a in enumerate(ARMS) if n_k[i] > 0})
        shares = bandit.allocate(ARMS, draw=DRAWS, floor=FLOOR, rng=rng).shares
        return np.tile([shares[a] for a in ARMS], (C, 1))

    return step


def orts_per_cell():
    """An independent OR-TS in each cell: no sharing between cells."""
    bandits, rng = [LogisticBandit() for _ in CELLS], np.random.default_rng(1)

    def step(n_t, s_t):
        weights = np.full((C, K), 1 / K)
        for c, bandit in enumerate(bandits):
            obs = {a: [n_t[c, i], s_t[c, i]] for i, a in enumerate(ARMS) if n_t[c, i] > 0}
            if len(obs) > 1 and 0 < s_t[c].sum() < n_t[c].sum():
                bandit.update(obs)
            if bandit.fitted:
                shares = bandit.allocate(ARMS, draw=DRAWS, floor=FLOOR, rng=rng).shares
                weights[c] = [shares[a] for a in ARMS]
        return weights

    return step


def orts_contextual(interaction_sd=None):
    def make():
        kwargs = {} if interaction_sd is None else {"interaction_sd": interaction_sd}
        bandit, rng = ContextualLogisticBandit(ARMS, list(range(C)), **kwargs), np.random.default_rng(1)

        def step(n_t, s_t):
            bandit.update({c: {a: [n_t[c, i], s_t[c, i]] for i, a in enumerate(ARMS)} for c in range(C)})
            out = bandit.allocate(draw=DRAWS, floor=FLOOR, rng=rng)
            return np.array([[out[c].shares[a] for a in ARMS] for c in range(C)])

        return step

    return make


REFERENCE = "Contextual OR-TS"
POLICIES = {
    REFERENCE: orts_contextual(),
    "GrowthBook contextual (as shipped)": gb_contextual,
    "GrowthBook non-contextual (as shipped)": gb_plain,
    "OR-TS, context ignored": orts_pooled,
    "OR-TS, independent per cell": orts_per_cell,
    "Contextual OR-TS, interaction_sd 0.02": orts_contextual(0.02),
    "Contextual OR-TS, interaction_sd 0.1": orts_contextual(0.1),
    "Contextual OR-TS, interaction_sd 0.25": orts_contextual(0.25),
    "Contextual OR-TS, interaction_sd 1.0": orts_contextual(1.0),
}


def run(make, h, sigma, seed):
    """Cumulative regret, share of users on their cell's best arm, and final leaf count."""
    env = np.random.default_rng(seed)
    region_effect = env.normal(0, h, (K, len(REGIONS))) if h else np.zeros((K, len(REGIONS)))
    device_effect = env.normal(0, h, (K, len(DEVICES))) if h else np.zeros((K, len(DEVICES)))
    contrast = np.array([MAIN + region_effect[:, r] + device_effect[:, d] for r, d in CELLS])
    base = BASE + env.normal(0, BASE_SD, C)
    step, weights = make(), np.full((C, K), 1 / K)
    regret, on_best = 0.0, 0.0
    for _ in range(T):
        p = 1 / (1 + np.exp(-(base[:, None] + env.normal(0, sigma) + contrast)))
        users = env.multinomial(N, CELL_SHARE)
        n_t = np.array([env.multinomial(users[c], weights[c]) for c in range(C)])
        s_t = env.binomial(n_t, p)
        regret += float((n_t * (p.max(1, keepdims=True) - p)).sum())
        on_best += float(n_t[np.arange(C), p.argmax(1)].sum())
        weights = step(n_t.astype(float), s_t.astype(float))
    leaves = step.leaves[0] if hasattr(step, "leaves") else np.nan
    if hasattr(step, "close"):
        step.close()
    return regret, on_best / (N * T), leaves


def main():
    global DRIVER
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", required=True, help="bundled gb_contextual_driver (driver.cjs)")
    parser.add_argument("--reps", type=int, default=20)
    args = parser.parse_args()
    DRIVER = args.driver
    scenarios = (
        (0.2, 0.0, "best arm differs by cell, no shifts"),
        (0.0, 0.0, "one best arm everywhere, no shifts"),
        (0.2, 0.3, "best arm differs by cell, common shifts sd 0.3"),
        (0.0, 0.3, "one best arm everywhere, common shifts sd 0.3"),
    )
    for h, sigma, label in scenarios:
        # the same seed gives every policy the same environment draw, so runs are paired
        res = {k: np.array([run(f, h, sigma, s) for s in range(args.reps)]) for k, f in POLICIES.items()}
        ref = res[REFERENCE]
        print(f"\n{label}: {T} periods x {N:,} users, {C} cells, {args.reps} paired runs", flush=True)
        for k, r in res.items():
            line = f"  {k:40} regret {r[:, 0].mean():7.1f}   on best arm {r[:, 1].mean():.3f}"
            if k != REFERENCE:
                d = r[:, 0] - ref[:, 0]
                se = d.std(ddof=1) / np.sqrt(args.reps)
                line += (f"   minus {REFERENCE} {d.mean():+7.1f} ± {se:5.1f} ({d.mean() / se:+.1f} SE),"
                         f" higher in {int((d > 0).sum())}/{args.reps}")
            if not np.isnan(r[:, 2]).all():
                line += f"   final leaves {r[:, 2].mean():.1f}"
            print(line, flush=True)


if __name__ == "__main__":
    main()
