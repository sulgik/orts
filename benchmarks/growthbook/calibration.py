"""Does gbstats report the right variance for the period-weighted mean of a binomial metric?

Run GrowthBook's bandit once with no shifts, keep the per-period allocation it produced, then
redraw the conversions many times on that fixed allocation. Compare the spread of each
variation's period-weighted mean with the variance gbstats reports for it (for a binomial metric,
the plain binomial variance, because `sum_squares = sum`).

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
    reported = (wm * (1 - wm) / (n_v - 1)).mean(0)
    actual = wm.var(0)
    print("arm   users     actual / reported variance")
    for k in range(g.K):
        print(f"  {k}  {n_v[k]:8,d}   {actual[k] / reported[k]:5.2f}")


if __name__ == "__main__":
    main()
