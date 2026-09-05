"""A migration and a stopping rule, end to end.

An incumbent Beta-Bernoulli service is replaced by OR-TS with a warm start
(paper, Supplement H), then run with the default stopping and dropping rule
of Section 6.1: drop an arm whose probability of being best stays below a
floor for several batches, stop when the leader's probability exceeds a
target and its expected loss is below what the business will forgo.

    python examples/ab_testing.py
"""
import numpy as np
from orts import LogisticBandit

rng = np.random.default_rng(8)
truth = {"control": 0.030, "variant_1": 0.031, "variant_2": 0.033}

# 1. The incumbent's per-arm Beta posteriors become the contrast prior.
incumbent = {"control": (1 + 90, 1 + 2910), "variant_1": (1 + 93, 1 + 2907), "variant_2": (1 + 99, 1 + 2901)}   # 3,000 exposures each so far
bandit = LogisticBandit.from_beta_posteriors(incumbent, reference="control")
alloc = {a: 1 / 3 for a in truth}

# 2. Run with the default rule.
DROP_BELOW, DROP_PATIENCE, STOP_ABOVE, LOSS_TOLERANCE = 0.01, 5, 0.95, 0.02
below = {a: 0 for a in truth}
active = list(truth)
for t in range(1, 41):
    level_shift = rng.normal(0, 0.25)                                     # the platform moves every day
    obs = {}
    for a in active:
        n = int(20000 * alloc[a])
        p = truth[a] * np.exp(level_shift) / (1 - truth[a] + truth[a] * np.exp(level_shift))
        obs[a] = [n, int(rng.binomial(n, p))]
    bandit.update(obs, remove_not_observed=True)
    q = bandit.query(active, draw=20000, rng=rng)
    alloc, loss, leader = q.shares, q.expected_loss, q.leader
    for a in active:
        below[a] = below[a] + 1 if alloc[a] < DROP_BELOW else 0
    dropped = [a for a in active if below[a] >= DROP_PATIENCE and a != leader]
    for a in dropped:
        active.remove(a)
        print(f"batch {t:2d}: drop {a} (P(best) < {DROP_BELOW} for {DROP_PATIENCE} batches)")
    if dropped:
        alloc = bandit.query(active, draw=20000, rng=rng).shares
    if alloc[leader] >= STOP_ABOVE and loss[leader] <= LOSS_TOLERANCE:
        print(f"batch {t:2d}: stop; {leader} has P(best) = {alloc[leader]:.3f}, expected loss {loss[leader]:.4f} log-odds")
        break
else:
    print("no stop within 40 batches; leader", leader, "at", round(alloc[leader], 3))
