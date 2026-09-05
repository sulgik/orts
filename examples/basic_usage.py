"""Algorithm 1 of the paper, one boundary at a time.

Run:  python examples/basic_usage.py
"""
import numpy as np
from orts import LogisticBandit

rng = np.random.default_rng(0)
bandit = LogisticBandit()                     # no prior information: flat on every coordinate

# Boundary 1.  The platform hands over the batch's counts {arm: [exposures, events]}.
batch = {"A": [30000, 300], "B": [30000, 330], "C": [30000, 290]}
bandit.update(batch)                          # R1 fit with a fresh flat intercept, R2 keep the contrasts
print("state carries:", bandit.get_models(), "(reference last)")
print("contrast means vs C:", np.round(bandit.mu[:-1], 3), " level (discarded next time):", round(bandit.mu[-1], 3))
q = bandit.query(["A", "B", "C"], draw=50000, rng=rng)                    # A1 draws, A2 winner shares
print("next allocation:", q.shares, " leader:", q.leader)

# Boundary 2.  The platform changed and every arm's rate halved; the contrasts did not move.
batch2 = {"A": [30000, 150], "B": [30000, 165], "C": [30000, 145]}
bandit.update(batch2)
print("after a level shift, contrast means:", np.round(bandit.mu[:-1], 3))
print("next allocation:", bandit.win_prop(draw=50000, rng=rng))

# The two controls of Section 5: decay acts on what is carried, aggressiveness on how strongly it is used.
bandit.update(batch2, decay=0.1)                                          # forget 10% of the carried precision
print("gamma = 2 concentrates:", bandit.query(aggressive=2.0, draw=50000, rng=rng).shares)
print("with a 5% floor:", bandit.query(floor=0.05, draw=50000, rng=rng).shares)

# The query's arm set is free: drop C from the next batch and add a brand-new arm D.
q = bandit.query(["A", "B", "D"], draw=50000, rng=rng)
print("query over A, B, D:", q.shares, "(D has no posterior yet and gets the uniform share)")

# Stopping-rule quantities (Section 6.1) come with every query.
print("P(best):", {a: round(v, 3) for a, v in q.p_best.items()})
print("expected loss of committing now:", {a: round(v, 4) for a, v in q.expected_loss.items()})
