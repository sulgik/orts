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

Coordinates.  The state ``(mu, sigma_inv)`` is kept in one canonical order,
``action_list``: arms in the order they were first seen, with the reference
arm last.  The first ``K-1`` entries are the log-odds contrasts of those
arms against the reference and the last entry is the level (the reference
arm's own log odds from the latest fit).  Batches may name arms in any
order and any subset; the canonical order never changes because of that.
If a batch does not expose the reference arm, the fit is done against an
observed arm and the result is mapped back.  Any subset and any new
reference is one linear map away (``get_par``; paper, Supplement B).
"""

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.linalg import inv, pinv

from .utils import estimate, is_pos_semidef

Obs = Dict[str, Sequence[float]]


@dataclass
class Allocation:
    """The answer to one action query: what to do with the next batch.

    ``arms`` are the arms the query named, in the caller's order.
    ``shares`` is the next batch's allocation over them, after the
    aggressiveness map and floors.  ``p_best`` is each arm's posterior
    probability of being best and ``expected_loss`` the expected loss, in
    log-odds units, of committing to that arm now; both are the raw
    Thompson quantities before shaping, and both are ``nan`` for an arm the
    state has never observed (it has no posterior; it gets the uniform
    share).  One set of Monte Carlo draws produces all three.
    """
    arms: List[str]
    shares: Dict[str, float]
    p_best: Dict[str, float] = field(default_factory=dict)
    expected_loss: Dict[str, float] = field(default_factory=dict)

    def __getitem__(self, arm: str) -> float:
        return self.shares[arm]

    @property
    def leader(self) -> str:
        return max(self.arms, key=lambda a: self.shares[a])


class LogisticBandit:
    """Odds-Ratio Thompson Sampling (default) and Full-TS.

    Parameters
    ----------
    mu, sigma_inv, action_list
        An existing state: contrast means, precision matrix, and arm names
        with the reference arm last.  Leave all three ``None`` to start from
        no prior information (zero precision, a flat prior on every
        coordinate).
    reference
        Which arm to hold as the canonical reference (for example the
        control).  It takes effect when that arm is first observed; until
        then, and by default, the first arm of the first batch is the
        reference.
    init_scale
        ``None`` (default) is the paper's basic specification: a flat
        contrast prior, under which the first fit requires every arm in the
        batch to have both events and non-events.  A positive ``tau`` selects
        the symmetric proper initialization of Supplement A, exchangeable arm
        effects ``N(0, tau^2)``, which gives every pairwise difference prior
        variance ``2 tau^2`` and every arm prior winner probability ``1/K``.
        It permits the first fit when individual arms have zero or complete
        counts, at the cost of a prespecified effect scale.  Choose it before
        observing outcomes; the paper says not to switch priors after
        separation is seen.
    new_arm_scale
        Prior sd for the contrast of an arm that joins after the first fit.
        ``None`` (default) is flat; a positive value is a proper ``N(0, s^2)``
        prior, an explicit option for arms whose first batch is separated.

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
                 action_list: Optional[List[str]] = None,
                 reference: Optional[str] = None,
                 init_scale: Optional[float] = None,
                 new_arm_scale: Optional[float] = None) -> None:
        if init_scale is not None and init_scale <= 0:
            raise ValueError("init_scale must be positive")
        if new_arm_scale is not None and new_arm_scale <= 0:
            raise ValueError("new_arm_scale must be positive")
        self.mu = np.array(mu, dtype=float) if mu is not None else np.array([])
        self.sigma_inv = (np.array(sigma_inv, dtype=float) if sigma_inv is not None
                          else np.empty((0, 0)))
        self.action_list = list(action_list) if action_list is not None else []
        # the arm to use as the canonical reference once it is first seen;
        # without it, the first arm of the first batch is the reference
        self._preferred_reference = reference
        self.init_scale = init_scale
        self.new_arm_scale = new_arm_scale
        # True once a fit has completed; before that the state is improper
        # (or absent) and queries return the start-up allocation
        self.fitted = len(self.action_list) > 0 and mu is not None

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
        by it.  The stored state is not modified.
        """
        if not action_list:
            return None, None
        for a in action_list:
            if a not in self.action_list:
                raise KeyError(f"arm '{a}' is not in the state")
        return _reexpress(self.mu, self.sigma_inv, self.action_list, list(action_list))

    def transform(self, action_list: List[str]) -> None:
        """Re-express the state in place for a subset or a new reference."""
        mu, sigma_inv = self.get_par(action_list)
        fitted = self.fitted
        self.__init__(mu=mu, sigma_inv=sigma_inv, action_list=action_list,
                      reference=self._preferred_reference, init_scale=self.init_scale,
                      new_arm_scale=self.new_arm_scale)
        self.fitted = fitted

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
            ``True`` if the state changed.  ``False`` if the batch was skipped
            and the state left as it was.  Two rules skip (paper, Algorithm 1
            and Supplement A).  Under the flat intercept prior a batch with no
            events, or with no non-events, has an improper posterior.  And a
            *first* fit under the flat contrast prior requires every arm in
            the batch to have both events and non-events; if that fails no
            state is formed, and ``query`` keeps returning the start-up
            allocation.  The paper says not to pool a skipped batch with later
            counts as though they shared one intercept, so callers should
            not add its counts to the next batch.
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
        first_fit = not self.fitted
        if first_fit and self.init_scale is None:
            # first-fit check: a flat contrast prior needs every arm to be
            # identified from this batch alone
            if any(v[1] == 0 or v[1] == v[0] for v in obs_valid.values()):
                return False

        # ---- canonical order after this batch: known non-reference arms in
        # first-seen order, then the new arms, then the reference last
        known = list(self.action_list)
        new = [a for a in obs_valid if a not in known]
        if not known:
            ref = (self._preferred_reference if self._preferred_reference in obs_valid
                   else next(iter(obs_valid)))
        else:
            ref = known[-1]
        if remove_not_observed and known:
            keep = [a for a in known if a in obs_valid]
            if keep:
                if ref not in keep:
                    ref = keep[-1]
                keep = [a for a in keep if a != ref] + [ref]
                self.transform(keep)
                known = list(self.action_list)
        canonical = [a for a in known if a != ref] + [a for a in new if a != ref] + [ref]

        # ---- fit layout: new arms, carried-but-unobserved arms, observed arms
        # with the fit reference last.  The fit reference is the canonical
        # reference when the batch exposes it, else the observed arm with the
        # most exposures; the level is that arm's log odds during the fit.
        unobserved = [a for a in known if a not in obs_valid]
        observed_known = [a for a in known if a in obs_valid]
        if ref in obs_valid:
            fit_ref = ref
        elif observed_known:
            fit_ref = max(observed_known, key=lambda a: obs_valid[a][0])
        else:
            fit_ref = max(new, key=lambda a: obs_valid[a][0])
        observed_known = [a for a in observed_known if a != fit_ref]
        new_fit = [a for a in new if a != fit_ref]
        fit_list = new_fit + unobserved + observed_known + [fit_ref]
        carried = unobserved + observed_known + ([fit_ref] if fit_ref in known else [])
        n_new = len(fit_list) - len(carried)

        prior = self.get_par(carried) if carried else (None, None)
        if prior[0] is not None and odds_ratios_only and len(prior[0]) > 1:
            # marginalize the level: keep the contrasts' covariance block and
            # give the new intercept zero precision (a flat prior)
            sigma_inv_new = np.zeros((len(prior[0]), len(prior[0])))
            sigma_inv_new[:-1, :-1] = inv(inv(prior[1])[:-1, :-1])
            prior = (prior[0], sigma_inv_new)
        if prior[0] is not None and fit_ref not in known:
            # the level coordinate of the carried state belongs to a
            # reference arm this batch does not expose; it is replaced by the
            # fit reference's level, which starts flat
            prior = (np.append(prior[0], 0.0), _pad_flat(prior[1]))

        # ---- prior over the full fit vector: new contrasts first
        K = len(fit_list)
        if first_fit and self.init_scale is not None:
            # symmetric proper initialization (Supplement A): S_0 = tau^-2 (I - 11'/K)
            # on the K-1 contrasts, zero precision on the level
            mu_full = np.zeros(K)
            S_full = np.zeros((K, K))
            S_full[:-1, :-1] = (np.eye(K - 1) - np.ones((K - 1, K - 1)) / K) / self.init_scale ** 2
        else:
            mu_full = np.zeros(K)
            S_full = np.zeros((K, K))
            if prior[0] is not None:
                mu_full[n_new:] = prior[0]
                S_full[n_new:, n_new:] = prior[1]
            if n_new and self.new_arm_scale is not None and not first_fit:
                S_full[:n_new, :n_new] = np.eye(n_new) / self.new_arm_scale ** 2
        prior_full = (mu_full, S_full)

        obs_list = [obs_valid[a] for a in new_fit + observed_known + [fit_ref]]
        indexes = [n_new, n_new + len(unobserved), K]
        mu, sigma_inv = estimate(prior_full, obs_list, indexes, discount=decay,
                                 prior_covers_all=True)
        self.fitted = True

        # ---- back to the canonical order
        self.mu, self.sigma_inv, self.action_list = mu, sigma_inv, fit_list
        if fit_list != canonical:
            self.mu, self.sigma_inv = _reexpress(self.mu, self.sigma_inv, fit_list, canonical)
            self.action_list = canonical
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
        # draw with the caller's last arm as the zero-scored reference; the
        # winner shares do not depend on which arm plays that role
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

    def query(self, arms: Optional[Sequence[str]] = None, draw: int = 100000,
              aggressive: float = 1.0, floor: float = 0.0,
              rng: Optional[np.random.Generator] = None) -> Allocation:
        """Steps A1-A2 as a query: name the arms that will be live in the next
        batch and get their allocation, plus the quantities a stopping rule reads.

        The arm set of a query need not match the state's.  Arms in the state
        but absent from the query are simply not allocated (their contrasts
        stay in memory); arms the state has never observed get the uniform
        share ``1/len(arms)``, since they have no posterior, and the observed
        arms share the rest in proportion to their winner probabilities.

        Parameters
        ----------
        arms
            The arms to allocate over, in the order the answer should use.
            ``None`` means every arm in the state.
        draw
            Monte Carlo size ``M``.
        aggressive
            ``gamma`` (paper, Section 5.2): winner shares are raised to this
            power and renormalized.  ``1`` is probability matching.
        floor
            Minimum share per arm, applied after the power map; the other
            arms are scaled to fill the remainder.
        """
        if draw <= 0:
            raise ValueError(f"draw must be positive, got {draw}")
        if aggressive <= 0:
            raise ValueError(f"aggressive must be positive, got {aggressive}")
        if not 0.0 <= floor < 1.0:
            raise ValueError(f"floor must be in [0, 1), got {floor}")
        arms = list(arms) if arms is not None else list(self.action_list)
        if len(set(arms)) != len(arms):
            raise ValueError("arms must be distinct")
        if not arms:
            return Allocation([], {}, {}, {})
        if not self.fitted:
            # no proper state yet: the start-up allocation (paper, Algorithm 1)
            nan = float("nan")
            return Allocation(arms, {a: 1.0 / len(arms) for a in arms},
                              {a: nan for a in arms}, {a: nan for a in arms})
        observed = [a for a in arms if a in self.action_list]
        unobserved = [a for a in arms if a not in self.action_list]
        nan = float("nan")
        shares: Dict[str, float] = {}
        p_best: Dict[str, float] = {a: nan for a in arms}
        loss: Dict[str, float] = {a: nan for a in arms}
        if len(observed) == 1:
            shares[observed[0]] = 1.0 / (1 + len(unobserved))
            p_best[observed[0]] = 1.0
            loss[observed[0]] = 0.0
        elif observed:
            names, scores = self.contrast_draws(observed, draw, rng)
            counts = np.bincount(scores.argmax(axis=1), minlength=len(names)).astype(float)
            raw = counts / counts.sum()
            l = (scores.max(axis=1, keepdims=True) - scores).mean(axis=0)
            shaped = counts ** aggressive
            p = shaped / shaped.sum()
            if floor > 0.0:
                p = _apply_floor(p, floor)
            share = len(observed) / float(len(observed) + len(unobserved))
            for i, a in enumerate(names):
                shares[a] = float(p[i] * share)
                p_best[a] = float(raw[i])
                loss[a] = float(l[i])
        for a in unobserved:
            shares[a] = 1.0 / float(len(observed) + len(unobserved))
        return Allocation(arms, {a: shares[a] for a in arms}, p_best, loss)

    def win_prop(self, action_list: Optional[List[str]] = None, draw: int = 100000,
                 aggressive: float = 1.0, floor: float = 0.0,
                 rng: Optional[np.random.Generator] = None) -> Dict[str, float]:
        """The next allocation as ``{arm: share}``; ``query(...).shares``."""
        return self.query(action_list, draw, aggressive, floor, rng).shares

    # ---------------------------------------------- stopping-rule quantities
    def expected_loss(self, action_list: Optional[List[str]] = None, draw: int = 100000,
                      rng: Optional[np.random.Generator] = None) -> Dict[str, float]:
        """``E[max_j beta_j - beta_i]`` per arm, in log-odds units; ``query(...).expected_loss``."""
        return self.query(action_list, draw, rng=rng).expected_loss

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

    def contrasts(self) -> Dict[str, Tuple[float, float]]:
        """``{arm: (mean, sd)}`` of each arm's log-odds contrast against the reference."""
        sds = self.contrast_sd()
        return {a: (float(self.mu[i]), sds[a]) for i, a in enumerate(self.action_list[:-1])}

    @property
    def level(self) -> Optional[float]:
        """The reference arm's log odds from the latest fit; not carried forward by OR-TS."""
        return float(self.mu[-1]) if len(self.mu) else None

    def set_reference(self, arm: str) -> None:
        """Re-base the canonical order on ``arm`` (a linear map; nothing is lost)."""
        if arm not in self.action_list:
            raise KeyError(f"arm '{arm}' is not in the state")
        self.transform([a for a in self.action_list if a != arm] + [arm])

    def drop(self, arms: Sequence[str]) -> None:
        """Fold the given arms out of the state (Supplement B); they can be
        re-introduced later by observing them, with a flat contrast prior."""
        keep = [a for a in self.action_list if a not in set(arms)]
        if not keep:
            self.__init__(reference=self._preferred_reference, init_scale=self.init_scale,
                          new_arm_scale=self.new_arm_scale)
            return
        ref = self.reference if self.reference in keep else keep[-1]
        self.transform([a for a in keep if a != ref] + [ref])

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


def _reexpress(mu: np.ndarray, sigma_inv: np.ndarray, old: List[str], new: List[str]
               ) -> Tuple[np.ndarray, np.ndarray]:
    """Map a state over ``old`` (reference last) to the coordinates of ``new``
    (its last arm as reference).  ``new`` must be a subset of ``old``."""
    new_nonref, new_ref = new[:-1], new[-1]
    to_logodds = np.eye(len(old))            # (contrasts, level) -> log odds of every arm
    to_logodds[:, -1] = 1.0
    to_new = np.zeros((len(new), len(old)))  # log odds -> (contrasts vs new_ref, new level)
    j_ref = old.index(new_ref)
    for i, arm in enumerate(new_nonref):
        to_new[i, old.index(arm)] = 1.0
        to_new[i, j_ref] = -1.0
    to_new[-1, j_ref] = 1.0
    T = to_new.dot(to_logodds)
    T_inv = np.rint(pinv(T))                 # entries are exactly 0 or +-1
    return T.dot(mu), T_inv.T.dot(sigma_inv).dot(T_inv)


def _pad_flat(sigma_inv: np.ndarray) -> np.ndarray:
    """Append one coordinate with zero precision (a flat prior)."""
    K = len(sigma_inv)
    out = np.zeros((K + 1, K + 1))
    out[:K, :K] = sigma_inv
    return out


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
