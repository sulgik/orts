"""OR-TS: Odds-Ratio Thompson Sampling for batched binary-reward bandits.

Reference implementation of the procedure described in

    S. Kim (2026). Odds-Ratio Thompson Sampling: A Specification and Design
    Guide for Contrast-Based Multi-Armed Bandits. arXiv:2609.19709.
    S. Kim and K. Kim (2020). Odds-ratio Thompson sampling to control for
    time-varying effect. arXiv:2003.01905.

Public API
----------
LogisticBandit   OR-TS (default) and Full-TS: a reference-coded logistic model
                 fitted once per batch with a fresh flat intercept; the state
                 carried across batches is the joint posterior of the log-odds
                 contrasts.  Decay, aggressiveness, allocation floors, changing
                 arm sets, warm starts and stopping-rule quantities are methods
                 or arguments on this class.  ``allocate(arms)`` is the action:
                 name the arms that will be live next and get an Allocation.
ContextualLogisticBandit
                 Experimental.  OR-TS over arm-by-cell contrasts: one logistic
                 model with a fresh flat intercept per cell and batch, cells
                 tied together by a hierarchical contrast prior.
allocate_from_rows, allocate_cells_from_rows, batches_from_rows, replay
                 From a table of per-period counts to the next allocation with
                 no state kept between runs; examples/sql/ has the query.
priors           The symmetric proper contrast prior the default starts from,
                 and Supplement B's augmentation for an arm that joins later.
Allocation       The answer to one action step: shares, P(best), expected loss.
TSPar            Beta-Bernoulli Thompson sampling, the per-arm baseline.
DiscountedTSPar  Beta-Bernoulli with geometric count discounting, the
                 forgetting baseline matched to OR-TS's decay.
diagnostics      Batch-level contrasts with sampling bands, excess variance,
                 the level-versus-contrast ratio R, and the implied decay.
"""

from .logisticbandit import LogisticBandit, Allocation
from .contextual import ContextualLogisticBandit
from .history import (allocate_from_rows, allocate_cells_from_rows, batches_from_rows,
                      replay)
from . import priors
from .ts import TSPar, DiscountedTSPar
from .utils import logistic, logit, estimate, is_pos_semidef
from . import diagnostics

__all__ = [
    "LogisticBandit", "ContextualLogisticBandit", "Allocation",
    "allocate_from_rows", "allocate_cells_from_rows", "batches_from_rows", "replay",
    "TSPar", "DiscountedTSPar",
    "logistic", "logit", "estimate", "is_pos_semidef", "diagnostics", "priors",
]
__version__ = "2.5.0"
