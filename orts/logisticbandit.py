"""OR-TS and Full-TS on a reference-coded logistic model.

One ``update`` is one boundary of the paper's Algorithm 1:

    R1  fit the reference-coded logistic model to the batch counts with a
        fresh flat-prior intercept and the carried contrast prior
        (scaled by ``1 - decay`` if decay is used);
    R2  keep the marginal contrast Gaussian ``(mu, S)`` and discard the
        intercept -- this is the OR-TS memory rule.  Full-TS keeps the
        joint posterior of intercept and contrasts instead.

and one ``win_prop`` is its action half:

    A1  draw contrast vectors, give the reference arm score 0, find the
        arm with the largest score in each draw;
    A2  turn the winner shares into the next allocation, optionally through
        the aggressiveness power map and allocation floors.

Coordinates.  The state ``(mu, sigma_inv)`` is over ``action_list``: the
first ``K-1`` entries are the log-odds contrasts of those arms against the
last one, and the last entry is the level (the reference arm's own log
odds).  Any subset and any new reference is one linear map away
(``get_par``; paper, Supplement B).
"""

import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.linalg import inv, pinv

from .utils import estimate, is_pos_semidef

Obs = Dict[str, Sequence[float]]


class LogisticBandit:
    """Odds-Ratio Thompson Sampling (default) and Full-TS.

    Parameters
    ----------
    mu, sigma_inv, action_list
        An existing state: contrast means, precision matrix, and arm names
        with the reference arm last.  Leave all three ``None`` to start from
        no prior information (zero precision, a flat prior on every
        coordinate).

    Examples
    --------
    >>> bandit = LogisticBandit()
    >>> bandit.update({"A": [30000, 300], "B": [30000, 290]})
    True
    >>> shares = bandit.win_prop(draw=20000)
    >>> sorted(shares) == ["A", "B"]
    True
    """

    def __init__(self, mu: Optional[np.ndarray] = None,
                 sigma_inv: Optional[np.ndarray] = None,
                 action_list: Optional[List[str]] = None) -> None:
        self.mu = np.array(mu, dtype=float) if mu is not None else np.array([])
        self.sigma_inv = (np.array(sigma_inv, dtype=float) if sigma_inv is not None
                          else np.empty((0, 0)))
        self.action_list = list(action_list) if action_list is not None else []

    # ------------------------------------------------------------------ state
    def get_models(self) -> List[str]:
        """Arms currently in the state, reference arm last."""
        return self.action_list

    @property
    def reference(self) -> Optional[str]:
        return self.action_list[-1] if self.action_list else None

    def get_par(self, action_list: Optional[List[str]] = None
                ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """The state re-expressed for ``action_list`` (its last arm as reference).

        This is the linear map of Supplement B: contrasts against a new
        reference are differences of contrasts against the old one, and the
        precision transforms accordingly.  Winner probabilities are unchanged
        by it.
        """
        if not action_list:
            return None, None
        for a in action_list:
            if a not in self.action_list:
                raise KeyError(f"arm '{a}' is not in the state")
        old = self.action_list
        new_nonref, new_ref = action_list[:-1], action_list[-1]
        # 1. from (contrasts vs old reference, level) to (log odds of every arm)
        to_logodds = np.eye(len(old))
        to_logodds[:, -1] = 1.0
        # 2. from log odds to (contrasts vs new reference, new level)
        to_new = np.zeros((len(action_list), len(old)))
        j_ref = old.index(new_ref)
        for i, arm in enumerate(new_nonref):
            to_new[i, old.index(arm)] = 1.0
            to_new[i, j_ref] = -1.0
        to_new[-1, j_ref] = 1.0
        T = to_new.dot(to_logodds)
        mu = T.dot(self.mu)
        T_inv = np.rint(pinv(T))          # entries are exactly 0/+-1
        sigma_inv = T_inv.T.dot(self.sigma_inv).dot(T_inv)
        return mu, sigma_inv

    def transform(self, action_list: List[str]) -> None:
        """Re-express the state in place for a subset or a new reference."""
        mu, sigma_inv = self.get_par(action_list)
        self.__init__(mu=mu, sigma_inv=sigma_inv, action_list=action_list)

    def covariance(self, action_list: Optional[List[str]] = None) -> np.ndarray:
        """Posterior covariance over ``action_list``'s coordinates."""
        _, sigma_inv = self.get_par(action_list or self.action_list)
        return pinv(sigma_inv)

    def contrast_sd(self) -> Dict[str, float]:
        """Posterior sd of each non-reference arm's contrast against the reference."""
        if len(self.action_list) < 2:
            return {}
        sd = np.sqrt(np.clip(np.diag(self.covariance()), 0.0, None))
        return {a: float(sd[i]) for i, a in enumerate(self.action_list[:-1])}

    # ------------------------------------------------------- recognition (R1-R2)
    def update(self, obs: Obs, odds_ratios_only: bool = True,
               remove_not_observed: bool = False, decay: float = 0.0) -> bool:
        """Absorb one batch of ``{arm: [exposures, events]}`` counts.

        Parameters
        ----------
        odds_ratios_only
            ``True`` is OR-TS: the level enters with a flat prior and is
            discarded after the fit.  ``False`` is Full-TS: the carried joint
            posterior of level and contrasts is the prior.
        remove_not_observed
            Drop arms that did not appear in this batch from the state
            (their contrasts are folded away by ``get_par``, not deleted from
            history; add them back later with a flat prior by observing them).
        decay
            ``lambda`` in ``[0, 1]``: the carried precision is scaled by
            ``1 - lambda`` before the fit (paper, Section 2.3).

        Returns
        -------
        bool
            ``True`` if the state changed.  ``False`` if the batch was skipped:
            under the flat intercept prior a batch with no events, or with no
            non-events, has an improper posterior and is skipped with the state
            left as it was (paper, Supplement A).
        """
        _validate(obs)
        if not 0.0 <= decay <= 1.0:
            raise ValueError("decay must be between 0.0 and 1.0, got {}".format(decay))

        obs_valid = {a: obs[a] for a in obs if obs[a][0] > 0}
        if not obs_valid:
            return False
        events = sum(v[1] for v in obs_valid.values())
        non_events = sum(v[0] - v[1] for v in obs_valid.values())
        if events == 0 or non_events == 0:
            return False

        known = self.get_models()
        action_on = [a for a in obs_valid if a in known]
        action_new = [a for a in obs_valid if a not in known]
        action_unobserved = [] if remove_not_observed else [a for a in known if a not in obs_valid]
        if remove_not_observed and known and action_on:
            self.transform([a for a in known if a in obs_valid])
        action_list = action_new + action_unobserved + action_on

        prior = self.get_par(action_unobserved + action_on)
        if prior[0] is not None and odds_ratios_only and len(prior[0]) > 1:
            # marginalize the level: keep the contrasts' covariance block,
            # give the new intercept zero precision (flat prior)
            sigma_inv_new = np.zeros((len(prior[0]), len(prior[0])))
            sigma_inv_new[:-1, :-1] = inv(inv(prior[1])[:-1, :-1])
            prior = (prior[0], sigma_inv_new)

        obs_list = [obs_valid[a] for a in action_new + action_on]
        indexes = [len(action_new), len(action_new) + len(action_unobserved), len(action_list)]
        mu, sigma_inv = estimate(prior, obs_list, indexes, discount=decay)
        self.mu, self.sigma_inv, self.action_list = mu, sigma_inv, action_list
        return True

    # ------------------------------------------------------------ action (A1-A2)
    def contrast_draws(self, action_list: Optional[List[str]] = None, draw: int = 100000,
                       rng: Optional[np.random.Generator] = None) -> Tuple[List[str], np.ndarray]:
        """Step A1's draws: ``draw`` contrast vectors with the reference arm's score 0.

        Returns ``(arms, scores)`` with ``scores`` of shape ``(draw, len(arms))``.
        """
        arms = list(action_list) if action_list else list(self.action_list)
        if draw <= 0:
            raise ValueError(f"draw must be positive, got {draw}")
        gen = rng if rng is not None else np.random.default_rng()
        if len(arms) == 1:
            return arms, np.zeros((draw, 1))
        mu, sigma_inv = self.get_par(arms)
        sigma = pinv(sigma_inv)[:-1, :-1]
        if len(arms) == 2:
            mc = gen.normal(mu[0], np.sqrt(max(sigma[0, 0], 0.0)), size=(draw, 1))
        else:
            if not is_pos_semidef(sigma, tol=1e-10):
                warnings.warn("contrast covariance not positive semidefinite; adding jitter",
                              RuntimeWarning)
                sigma = sigma + 1e-6 * np.eye(len(sigma))
            mc = gen.multivariate_normal(mu[:-1], sigma, size=draw, method="svd")
        return arms, np.concatenate((mc, np.zeros((draw, 1))), axis=1)

    def win_prop(self, action_list: Optional[List[str]] = None, draw: int = 100000,
                 aggressive: float = 1.0, floor: float = 0.0,
                 rng: Optional[np.random.Generator] = None) -> Dict[str, float]:
        """Step A2: the next allocation as probability matching, optionally shaped.

        Parameters
        ----------
        action_list
            Arms to allocate over.  Arms not yet in the state get the uniform
            share ``1/len(action_list)`` (a brand-new arm has no posterior),
            and the observed arms share the rest in proportion to their
            winner probabilities.
        draw
            Monte Carlo size ``M``.
        aggressive
            ``gamma`` of the paper's Section 5.2: winner shares are raised to
            this power and renormalized.  ``1`` is probability matching,
            ``>1`` concentrates, ``<1`` flattens.
        floor
            Minimum share per arm, applied after the power map and followed
            by renormalization.
        """
        if aggressive <= 0:
            raise ValueError(f"aggressive must be positive, got {aggressive}")
        if not 0.0 <= floor < 1.0:
            raise ValueError(f"floor must be in [0, 1), got {floor}")
        arms = list(action_list) if action_list is not None else list(self.action_list)
        if not arms:
            return {}
        observed = [a for a in arms if a in self.action_list]
        unobserved = [a for a in arms if a not in self.action_list]
        out: Dict[str, float] = {}
        if len(observed) == 1:
            out[observed[0]] = 1.0 / (1 + len(unobserved))
        elif observed:
            names, scores = self.contrast_draws(observed, draw, rng)
            counts = np.bincount(scores.argmax(axis=1), minlength=len(names)).astype(float)
            shaped = counts ** aggressive
            p = shaped / shaped.sum()
            if floor > 0.0:
                p = _apply_floor(p, floor)
            share = len(observed) / float(len(observed) + len(unobserved))
            for i, a in enumerate(names):
                out[a] = float(p[i] * share)
        for a in unobserved:
            out[a] = 1.0 / float(len(observed) + len(unobserved))
        return out

    # ---------------------------------------------- stopping-rule quantities
    def expected_loss(self, action_list: Optional[List[str]] = None, draw: int = 100000,
                      rng: Optional[np.random.Generator] = None) -> Dict[str, float]:
        """``E[max_j beta_j - beta_i]`` per arm, in log-odds units.

        The expected loss of committing to arm ``i`` now; with ``win_prop``
        it is what a stopping or dropping rule reads (paper, Section 6.1).
        """
        names, scores = self.contrast_draws(action_list, draw, rng)
        loss = (scores.max(axis=1, keepdims=True) - scores).mean(axis=0)
        return {a: float(loss[i]) for i, a in enumerate(names)}

    def implied_decay(self, excess_sd_beta: float) -> float:
        """``lambda = tau^2 / (v + tau^2)``: the decay a measured contrast drift implies.

        ``tau`` is the per-batch excess sd of the contrast (``orts.diagnostics``
        on logged counts; paper, Section 4.1) and ``v`` the median posterior
        variance of the current contrasts (paper, Supplement H).
        """
        if excess_sd_beta < 0:
            raise ValueError("excess_sd_beta must be non-negative")
        sds = self.contrast_sd()
        if not sds:
            return 0.0
        v = float(np.median([s * s for s in sds.values()]))
        tau2 = float(excess_sd_beta) ** 2
        return 0.0 if tau2 == 0.0 else tau2 / (v + tau2)

    # ------------------------------------------------------------- warm start
    @classmethod
    def from_beta_posteriors(cls, posteriors: Dict[str, Sequence[float]],
                             reference: Optional[str] = None) -> "LogisticBandit":
        """Warm start from an incumbent Beta-Bernoulli service's ``{arm: (a, b)}``.

        Each arm's log odds is approximately ``N(log(a/b), 1/a + 1/b)``
        (the logit-normal approximation), so the contrasts against the
        reference inherit means ``log(a_i/b_i) - log(a_K/b_K)`` and the
        shared-reference covariance ``v_i + v_K`` on the diagonal and ``v_K``
        off it (paper, Supplement H).  The level's belief is carried too, but
        the first OR-TS update discards it, which is the point of migrating.
        """
        arms = list(posteriors)
        if len(arms) < 2:
            raise ValueError("a warm start needs at least two arms")
        ref = reference if reference is not None else arms[-1]
        if ref not in arms:
            raise KeyError(f"reference '{ref}' not among the arms")
        order = [a for a in arms if a != ref] + [ref]
        a = np.array([float(posteriors[x][0]) for x in order])
        b = np.array([float(posteriors[x][1]) for x in order])
        if np.any(a <= 0) or np.any(b <= 0):
            raise ValueError("Beta parameters must be positive")
        m, v = np.log(a / b), 1.0 / a + 1.0 / b
        K = len(order)
        mu = np.append(m[:-1] - m[-1], m[-1])
        cov = np.full((K, K), v[-1])
        cov[:-1, :-1] += np.diag(v[:-1])
        cov[:-1, -1] = cov[-1, :-1] = -v[-1]
        cov[-1, -1] = v[-1]
        return cls(mu=mu, sigma_inv=inv(cov), action_list=order)


def _apply_floor(p: np.ndarray, floor: float) -> np.ndarray:
    """Give every arm at least ``floor`` and scale the others to fill the rest."""
    if floor * len(p) > 1.0:
        raise ValueError(f"floor {floor} is infeasible for {len(p)} arms")
    p = np.asarray(p, dtype=float).copy()
    fixed = np.zeros(len(p), dtype=bool)
    for _ in range(len(p)):
        low = (p < floor) & ~fixed
        if not low.any():
            break
        fixed |= low
        free = ~fixed
        remaining = 1.0 - floor * fixed.sum()
        total_free = p[free].sum()
        p[fixed] = floor
        p[free] = remaining * p[free] / total_free if total_free > 0 else remaining / max(free.sum(), 1)
    return p


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
