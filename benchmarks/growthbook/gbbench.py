"""OR-TS against GrowthBook's bandit engine, on binomial metrics.

GrowthBook's bandit runs in two stages. SQL (`bandit-statistics-cte.ts`) reduces each
variation's per-period data to one row, and `gbstats` turns that row into allocation weights
(`BanditsSimple.compute_result`). This script rebuilds the SQL reduction in Python and feeds the
row to the real `BanditsSimple`, with the settings the product passes in
(`BanditSettingsForStatsEngine`: `weight_by_period=True`, `top_two=False`,
`min_variation_weight=0.01`, `bandit_weights_seed=100`) and the prior of `preprocess_bandits`.

The SQL reduction, for variation v over periods p:

    weight_p         = users in period p (all variations) / all users
    main_sum         = N_v * sum_p weight_p * mean_{v,p}
    main_sum_squares = N_v * (sum_p weight_p^2 * var_{v,p} / n_{v,p}) * (N_v - 1)
                       + N_v * (sum_p weight_p * mean_{v,p})^2

For a binomial metric `base_statistic_from_metric_row` builds a `ProportionStatistic` from
`main_sum` and the count only, so `main_sum_squares` is not used: `create_bandit_statistics` recasts
it with `sum_squares = sum`. The "binomial" policy below follows that path; the "count" policy uses
the SQL variance instead, as a diagnostic.

Environment: K arms whose log-odds differ by fixed contrasts, all moved each period by a common
shift (sd `sigma`); `sigma = 0` is the control. Each period serves N users by the current weights.

    python gbbench.py --reps 20
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
from gbstats.bayesian.bandits import BanditConfig, BanditsSimple
from gbstats.bayesian.tests import GaussianPrior
from gbstats.models.statistics import SampleMeanStatistic

from orts import LogisticBandit

warnings.filterwarnings("ignore")

K = 5
CONTRASTS = np.array([0.0, 0.05, 0.10, 0.15, 0.20])  # log-odds; the last arm is best
BASE = np.log(0.03 / 0.97)  # 3% base rate
N, T = 20_000, 40  # users per period, periods
FLOOR = 0.01  # GrowthBook's min_variation_weight, used for OR-TS too
ARMS = [f"a{i}" for i in range(K)]


def gb_row(n: np.ndarray, s: np.ndarray, weight_by_period: bool, binomial: bool):
    """The statistics BanditsSimple receives. n, s: (periods, K) users and conversions."""
    n_v = n.sum(0)
    if not weight_by_period:
        return [SampleMeanStatistic(n=int(n_v[k]), sum=float(s[:, k].sum()),
                                    sum_squares=float(s[:, k].sum())) for k in range(K)]
    w = n.sum(1) / n.sum()  # the same period weights for every variation
    out = []
    for k in range(K):
        ok = n[:, k] > 0  # SQL groups by (variation, period): absent periods drop out
        m = np.where(ok, s[:, k] / np.maximum(n[:, k], 1), 0.0)
        var = np.where(n[:, k] > 1, m * (1 - m) * n[:, k] / np.maximum(n[:, k] - 1, 1), 0.0)
        wm = float((w[ok] * m[ok]).sum())
        if binomial:
            out.append(SampleMeanStatistic(n=int(n_v[k]), sum=wm * n_v[k], sum_squares=wm * n_v[k]))
        else:
            se2 = float((w[ok] ** 2 * var[ok] / np.maximum(n[ok, k], 1)).sum())
            out.append(SampleMeanStatistic(n=int(n_v[k]), sum=wm * n_v[k],
                                           sum_squares=n_v[k] * (se2 * (n_v[k] - 1) + wm ** 2)))
    return out


def growthbook(weight_by_period=True, binomial=True):
    cfg = BanditConfig(prior_distribution=GaussianPrior(mean=0, variance=1e4, proper=True),
                       weight_by_period=weight_by_period, top_two=False,
                       min_variation_weight=FLOOR, bandit_weights_seed=100)
    ns, ss, alloc = [], [], np.full(K, 1 / K)

    def step(n_t, s_t):
        nonlocal alloc
        ns.append(n_t)
        ss.append(s_t)
        stats = gb_row(np.array(ns), np.array(ss), weight_by_period, binomial)
        result = BanditsSimple(stats, list(alloc), cfg).compute_result()
        if result.bandit_weights is not None:  # None until every variation has 100 users
            alloc = np.array(result.bandit_weights)
        return alloc

    return step


def orts():
    bandit = LogisticBandit()
    rng = np.random.default_rng(1)

    def step(n_t, s_t):
        bandit.update({a: [int(n_t[i]), int(s_t[i])] for i, a in enumerate(ARMS) if n_t[i] > 0})
        shares = bandit.allocate(ARMS, draw=20_000, floor=FLOOR, rng=rng).shares
        return np.array([shares[a] for a in ARMS])

    return step


POLICIES = {
    "OR-TS": orts,
    "GrowthBook as shipped (binomial, period-weighted)": growthbook,
    "GrowthBook unweighted (pooled)": lambda: growthbook(weight_by_period=False),
    "GrowthBook period-weighted, SQL variance [diagnostic]": lambda: growthbook(binomial=False),
}


def run(make, sigma, seed):
    """Cumulative regret and share of users on the best arm, for one environment draw."""
    rng = np.random.default_rng(seed)
    step, alloc = make(), np.full(K, 1 / K)
    regret, on_best = 0.0, 0
    for _ in range(T):
        p = 1 / (1 + np.exp(-(BASE + rng.normal(0, sigma) + CONTRASTS)))
        n_t = rng.multinomial(N, alloc)
        s_t = rng.binomial(n_t, p)
        regret += float((n_t * (p.max() - p)).sum())
        on_best += n_t[-1]
        alloc = step(n_t, s_t)
    return regret, on_best / (N * T)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=20)
    args = parser.parse_args()
    for sigma, label in ((0.3, "common shifts, sd 0.3"), (0.0, "no shifts (control)")):
        # the same seed gives every policy the same environment draw, so runs are paired
        res = {k: np.array([run(f, sigma, s) for s in range(args.reps)]) for k, f in POLICIES.items()}
        ref = res["OR-TS"]
        print(f"\n{label}: {T} periods x {N:,} users, {args.reps} paired runs")
        for k, r in res.items():
            line = f"  {k:54} regret {r[:, 0].mean():7.1f}   best arm {r[:, 1].mean():.3f}"
            if k != "OR-TS":
                d = r[:, 0] - ref[:, 0]
                se = d.std(ddof=1) / np.sqrt(args.reps)
                line += (f"   minus OR-TS {d.mean():+7.1f} ± {se:5.1f} ({d.mean() / se:+.1f} SE),"
                         f" median {np.median(d):+6.1f}, OR-TS lower in {int((d > 0).sum())}/{args.reps}")
            print(line)


if __name__ == "__main__":
    main()
