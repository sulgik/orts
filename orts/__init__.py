"""OR-TS: Odds-Ratio Thompson Sampling for batched binary-reward bandits.

Reference implementation of the procedure described in

    S. Kim (2026). Odds-Ratio Thompson Sampling: A Specification and Design
    Guide for Contrast-Based Multi-Armed Bandits.
    S. Kim and K. Kim (2020). Odds-ratio Thompson sampling to control for
    time-varying effect. arXiv:2003.01905.

Public API
----------
LogisticBandit   OR-TS (default) and Full-TS: a reference-coded logistic model
                 fitted once per batch with a fresh flat intercept; the state
                 carried across batches is the joint posterior of the log-odds
                 contrasts.  Decay, aggressiveness, allocation floors, changing
                 arm sets, warm starts and stopping-rule quantities are methods
                 or arguments on this class.  ``query(arms)`` is the action:
                 name the arms that will be live next and get an Allocation.
Allocation       The answer to a query: shares, P(best), expected loss.
TSPar            Beta-Bernoulli Thompson sampling, the per-arm baseline.
DiscountedTSPar  Beta-Bernoulli with geometric count discounting, the
                 forgetting baseline matched to OR-TS's decay.
diagnostics      Batch-level contrasts with sampling bands, excess variance,
                 the level-versus-contrast ratio R, and the implied decay.
"""

from .logisticbandit import LogisticBandit, Allocation
from .ts import TSPar, DiscountedTSPar
from .utils import logistic, logit, estimate, is_pos_semidef
from . import diagnostics

__all__ = [
    "LogisticBandit", "Allocation", "TSPar", "DiscountedTSPar",
    "logistic", "logit", "estimate", "is_pos_semidef", "diagnostics",
]
__version__ = "2.1.0"
