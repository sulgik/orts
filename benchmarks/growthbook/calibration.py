"""How wrong would the plain binomial variance be for the period-weighted mean?

Run GrowthBook's bandit once with no shifts, keep the per-period allocation it produced, then
redraw the conversions many times on that fixed allocation. Compare the spread of each
variation's period-weighted mean with the plain binomial variance `p(1-p)/n` of the same mean.

This is a what-if, not a description of the product. GrowthBook changes a binomial metric's type
to count before the bandit engine runs (`process_experiment_results` in gbstats), so the variance
it actually uses is the SQL's `main_sum_squares`, which is the variance of the weighted mean. An
earlier version of this script presented the plain binomial variance as what gbstats reports.

    python calibration.py
"""
import numpy as np

import gbbench as g


def main(seed=3, draws=4000):
    rng = np.random.default_rng(seed)
    p = 1 / (1 + np.exp(-(g.BASE + g.CONTRASTS)))
    step, alloc, ns = g.growthbook(), np.full(g.K, 1 / g.K), []
    for _ in range(g.T):
        n_t = rng.multinomial(g.N, alloc)
        ns.append(n_t)
        alloc = step(n_t, rng.binomial(n_t, p))
    n = np.array(ns)
    w = n.sum(1) / n.sum()
    m = rng.binomial(n[None], p, size=(draws,) + n.shape) / np.maximum(n, 1)
    wm = (w[None, :, None] * m).sum(1)
    n_v = n.sum(0)
    reported = (wm * (1 - wm) / (n_v - 1)).mean(0)  # the plain binomial variance of the mean
    actual = wm.var(0)
    print("arm   users     actual / plain-binomial variance")
    for k in range(g.K):
        print(f"  {k}  {n_v[k]:8,d}   {actual[k] / reported[k]:5.2f}")


if __name__ == "__main__":
    main()
