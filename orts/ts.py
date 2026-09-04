"""Beta-Bernoulli Thompson sampling: the per-arm baseline and its discounted variant."""

from typing import Dict, List, Optional, Sequence

import numpy as np


def _validate(obs) -> None:
    if not obs:
        raise ValueError("obs dictionary cannot be empty")
    for action, values in obs.items():
        if not isinstance(values, (list, tuple, np.ndarray)) or len(values) != 2:
            raise ValueError(
                "Each observation must be [total_count, success_count], "
                "got {} for action '{}'".format(values, action))
        total, success = values
        if total < 0:
            raise ValueError("Total count must be non-negative, got {} for action '{}'".format(total, action))
        if success < 0:
            raise ValueError("Success count must be non-negative, got {} for action '{}'".format(success, action))
        if success > total:
            raise ValueError("Success count ({}) cannot exceed total count ({}) for action '{}'".format(
                success, total, action))


class TSPar:
    """Beta-Bernoulli Thompson sampling with independent Beta(1, 1) priors.

    The state is two counters per arm.  Across batches it carries each arm's
    absolute event rate: the per-arm bet of the paper's Section 3.
    """

    def __init__(self) -> None:
        self.action_list: List[str] = []
        self.alpha = np.array([], dtype=float)
        self.beta = np.array([], dtype=float)

    def get_models(self) -> List[str]:
        return self.action_list

    def update(self, obs: Dict[str, Sequence[float]], **_ignored) -> None:
        _validate(obs)
        for action in obs:
            if action not in self.action_list:
                self.action_list.append(action)
                self.alpha = np.append(self.alpha, 1.0)
                self.beta = np.append(self.beta, 1.0)
        for i, action in enumerate(self.action_list):
            if action in obs:
                n, c = obs[action]
                self.alpha[i] += c
                self.beta[i] += n - c

    def draws(self, draw: int = 10000, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        gen = rng if rng is not None else np.random.default_rng()
        return gen.beta(self.alpha, self.beta, size=(draw, len(self.action_list)))

    def win_prop(self, draw: int = 10000, rng: Optional[np.random.Generator] = None) -> Dict[str, float]:
        if draw <= 0:
            raise ValueError(f"draw must be positive, got {draw}")
        if not self.action_list:
            return {}
        winners = self.draws(draw, rng).argmax(axis=1)
        counts = np.bincount(winners, minlength=len(self.action_list)) / float(draw)
        return {a: float(counts[i]) for i, a in enumerate(self.action_list)}


class DiscountedTSPar(TSPar):
    """Beta-Bernoulli TS with geometric discounting toward the Beta(1, 1) prior.

    Before absorbing a batch both counters shrink by ``1 - discount`` around
    the prior, so evidence from ``k`` batches ago carries weight
    ``(1 - discount)^k``.  Tempering a Beta density is exactly this count
    discounting, so ``discount`` here and ``decay`` in ``LogisticBandit``
    carry the same effective memory of roughly ``1/discount`` batches
    (paper, Section 2.3; Raj and Kalyani 2017).
    """

    def __init__(self, discount: float) -> None:
        if not 0.0 <= discount <= 1.0:
            raise ValueError(f"discount must be in [0, 1], got {discount}")
        super().__init__()
        self.discount = float(discount)

    def update(self, obs: Dict[str, Sequence[float]], **_ignored) -> None:
        if self.action_list:
            self.alpha = 1.0 + (1.0 - self.discount) * (self.alpha - 1.0)
            self.beta = 1.0 + (1.0 - self.discount) * (self.beta - 1.0)
        super().update(obs)
