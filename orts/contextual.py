"""Contextual OR-TS: one logistic model over arm-by-cell contrasts.  Experimental.

A context is a discrete *cell* (a segment, a tree leaf, a cross of attribute
levels).  Each batch is fitted with

    logit p[t, c, a] = alpha[t, c] + theta[a, c],      theta[reference, c] = 0,

where ``alpha[t, c]`` is a fresh flat-prior intercept for cell ``c`` in batch
``t`` and ``theta[a, c]`` is arm ``a``'s log-odds contrast against the
reference arm in cell ``c``.  As in ``LogisticBandit`` the memory rule is
OR-TS's: keep the joint Gaussian of the contrasts, discard every intercept.
A cell's base rate and any shift common to its arms are therefore never
carried from one batch to the next.

What ties the cells together is the prior.  With latent arm effects
``z[a] ~ N(0, tau^2)`` and latent arm-by-cell effects ``u[a, c] ~ N(0, omega^2)``,
all independent, ``theta[a, c] = (z[a] - z[ref]) + (u[a, c] - u[ref, c])`` and

    Cov(theta) = (tau^2 J + omega^2 I)  (x)  (I + 11'),

``J`` the all-ones matrix over cells and ``(I + 11')`` the symmetric contrast
covariance of ``orts.priors``.  ``omega`` (``interaction_sd``) is the one knob:
as it goes to zero every cell shares one contrast vector, which is the
non-contextual OR-TS; as it grows the cells decouple into independent OR-TS
runs.  In between, a thin cell borrows from the others in proportion to how
little it knows.  With a single cell the model is ``LogisticBandit`` with
``arm_effect_prior_sd = sqrt(tau^2 + omega^2)``.

How much the cells really differ is not something to guess, so by default
``omega`` is estimated (``interaction_sd="auto"``).  The state keeps what the
batches have said about the contrasts apart from the prior, as a Gaussian
likelihood ``exp(-theta' L theta / 2 + b' theta)``.  After each batch
``omega`` is set to the value on a grid that maximizes the marginal likelihood

    log m(omega) = -1/2 log|I + Sigma_0(omega) L| + 1/2 b' (Sigma_0(omega)^-1 + L)^-1 b,

and the posterior is re-formed under it.  Cells whose contrasts agree pull
``omega`` down and pool; cells that disagree push it up and separate.  Nothing
is lost by pooling early, because ``L`` and ``b`` are kept whole.

Not covered yet: decay, arms or cells that join after construction, and
continuous covariates.
"""

import math
from typing import Dict, Hashable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .logisticbandit import DEFAULT_ARM_EFFECT_PRIOR_SD, Allocation, _apply_floor

#: Where ``interaction_sd="auto"`` starts, before any batch has been seen.
DEFAULT_INTERACTION_SD = 0.25

#: The values ``interaction_sd="auto"`` chooses among, in log-odds units.
INTERACTION_SD_GRID = np.geomspace(0.01, 1.0, 21)

CellObs = Dict[Hashable, Dict[str, Sequence[float]]]


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value}")
    return value


class ContextualLogisticBandit:
    """OR-TS over a fixed set of arms and context cells.

    Parameters
    ----------
    arms
        The arms; the last one is the reference.  The prior is symmetric, so
        the choice does not change any pairwise posterior.
    cells
        The context cells, any hashable labels.
    arm_effect_prior_sd
        ``tau``: sd of the latent arm effects shared by every cell.
    interaction_sd
        ``omega``: sd of the latent arm-by-cell effects.  Smaller values pool
        the cells harder.  ``"auto"`` (the default) re-estimates it after
        every batch by marginal likelihood over ``INTERACTION_SD_GRID``; a
        number fixes it.  The current value is the ``interaction_sd`` attribute.
    """

    def __init__(self, arms: Sequence[str], cells: Sequence[Hashable],
                 arm_effect_prior_sd: float = DEFAULT_ARM_EFFECT_PRIOR_SD,
                 interaction_sd: Union[float, str] = "auto") -> None:
        self.arms: List[str] = list(arms)
        self.cells: List[Hashable] = list(cells)
        if len(self.arms) < 2 or len(set(self.arms)) != len(self.arms):
            raise ValueError("arms must be at least two distinct names")
        if not self.cells or len(set(self.cells)) != len(self.cells):
            raise ValueError("cells must be one or more distinct labels")
        self.arm_effect_prior_sd = _positive(arm_effect_prior_sd, "arm_effect_prior_sd")
        self.estimates_interaction_sd = isinstance(interaction_sd, str)
        if self.estimates_interaction_sd:
            if interaction_sd != "auto":
                raise ValueError(f'interaction_sd must be a positive number or "auto", got {interaction_sd!r}')
            interaction_sd = DEFAULT_INTERACTION_SD
        self.interaction_sd = _positive(interaction_sd, "interaction_sd")
        self._arm_index = {a: i for i, a in enumerate(self.arms)}
        self._cell_index = {c: i for i, c in enumerate(self.cells)}
        K, C = len(self.arms), len(self.cells)
        self._k = K - 1  # contrasts per cell
        #: posterior mean and precision of the contrasts, cell-major:
        #: coordinate ``c * (K - 1) + a`` is arm ``a`` against the reference in cell ``c``
        self.mu = np.zeros(C * self._k)
        self.precision = self._prior_precision(self.interaction_sd)
        # what the batches have said, apart from the prior: exp(-theta' L theta / 2 + b' theta)
        self._data_precision = np.zeros_like(self.precision)
        self._data_shift = np.zeros_like(self.mu)
        self.fitted = False

    def _prior_precision(self, interaction_sd: float) -> np.ndarray:
        """Inverse of ``(tau^2 J + omega^2 I) (x) (I + 11')``."""
        K, C = len(self.arms), len(self.cells)
        across = self.arm_effect_prior_sd ** 2 * np.ones((C, C)) + interaction_sd ** 2 * np.eye(C)
        within_precision = np.eye(self._k) - np.ones((self._k, self._k)) / K
        return np.kron(np.linalg.inv(across), within_precision)

    def _log_marginal(self, interaction_sd: float) -> float:
        """Log marginal likelihood of the batches so far under ``interaction_sd``, up to a constant."""
        prior = self._prior_precision(interaction_sd)
        posterior = prior + self._data_precision
        return float(0.5 * (np.linalg.slogdet(prior)[1] - np.linalg.slogdet(posterior)[1])
                     + 0.5 * self._data_shift @ np.linalg.solve(posterior, self._data_shift))

    # ------------------------------------------------------------------ update
    def update(self, obs: CellObs) -> bool:
        """Absorb one batch of ``{cell: {arm: [exposures, events]}}`` counts.

        A cell may be absent and an arm may be absent from a cell.  A cell
        whose batch has no events, or nothing but events, says nothing about
        its contrasts once its intercept is free, so it is skipped.  Returns
        ``False`` if no cell was informative and the state is unchanged.
        """
        rows = []  # (cell position among the fitted cells, theta coordinate or -1, n, s)
        fitted_cells = 0
        for cell, arms in obs.items():
            if cell not in self._cell_index:
                raise KeyError(f"unknown cell {cell!r}")
            c = self._cell_index[cell]
            cell_rows = []
            for arm, (n, s) in arms.items():
                if arm not in self._arm_index:
                    raise KeyError(f"unknown arm {arm!r}")
                n, s = float(n), float(s)
                if n < 0 or s < 0 or s > n:
                    raise ValueError(f"need 0 <= events <= exposures, got {[n, s]} for {arm!r} in {cell!r}")
                if n == 0:
                    continue
                a = self._arm_index[arm]
                coord = c * self._k + a if a < self._k else -1
                cell_rows.append((coord, n, s))
            events = sum(r[2] for r in cell_rows)
            exposures = sum(r[1] for r in cell_rows)
            if len(cell_rows) < 2 or events == 0 or events == exposures:
                continue
            rows += [(fitted_cells, coord, n, s) for coord, n, s in cell_rows]
            fitted_cells += 1
        if not rows:
            return False

        P = len(self.mu)
        level = P + np.array([r[0] for r in rows])
        coord = np.array([r[1] for r in rows])
        n = np.array([r[2] for r in rows])
        s = np.array([r[3] for r in rows])
        has_theta = coord >= 0
        # the design: one intercept column per fitted cell after the P contrasts
        X = np.zeros((len(rows), P + fitted_cells))
        X[np.arange(len(rows)), level] = 1.0
        X[np.flatnonzero(has_theta), coord[has_theta]] = 1.0
        prior_precision = np.zeros((P + fitted_cells, P + fitted_cells))
        prior_precision[:P, :P] = self.precision
        prior_mean = np.concatenate([self.mu, np.zeros(fitted_cells)])

        def objective(w: np.ndarray) -> float:
            eta = X @ w
            d = w - prior_mean
            return float(np.sum(n * np.logaddexp(0.0, eta) - s * eta) + 0.5 * d @ prior_precision @ d)

        # start each intercept at its cell's pooled log odds, contrasts at the prior mean
        w = prior_mean.copy()
        for j in range(fitted_cells):
            in_cell = level == P + j
            rate = s[in_cell].sum() / n[in_cell].sum()
            w[P + j] = math.log(rate / (1.0 - rate))
        value = objective(w)
        for _ in range(100):
            p = 1.0 / (1.0 + np.exp(-(X @ w)))
            gradient = X.T @ (n * p - s) + prior_precision @ (w - prior_mean)
            hessian = (X * (n * p * (1.0 - p))[:, None]).T @ X + prior_precision
            step = np.linalg.solve(hessian, gradient)
            scale = 1.0
            while True:  # backtrack: Newton can overshoot when a cell has few events
                candidate = w - scale * step
                candidate_value = objective(candidate)
                if candidate_value <= value or scale < 1e-8:
                    break
                scale *= 0.5
            converged = abs(value - candidate_value) < 1e-10 * (1.0 + abs(value))
            w, value = candidate, candidate_value
            if converged:
                break

        p = 1.0 / (1.0 + np.exp(-(X @ w)))
        hessian = (X * (n * p * (1.0 - p))[:, None]).T @ X + prior_precision
        # keep the contrasts' marginal Gaussian: the Schur complement drops the intercepts
        cross = hessian[:P, P:]
        self.precision = hessian[:P, :P] - cross @ np.linalg.solve(hessian[P:, P:], cross.T)
        self.precision = 0.5 * (self.precision + self.precision.T)
        self.mu = w[:P].copy()
        # the prior has mean zero, so the posterior's natural parameters less the
        # prior's are the batches' own
        self._data_precision = self.precision - self._prior_precision(self.interaction_sd)
        self._data_shift = self.precision @ self.mu
        if self.estimates_interaction_sd:
            self.interaction_sd = float(max(INTERACTION_SD_GRID, key=self._log_marginal))
            self.precision = self._prior_precision(self.interaction_sd) + self._data_precision
            self.mu = np.linalg.solve(self.precision, self._data_shift)
        self.fitted = True
        return True

    # ------------------------------------------------------------------- state
    def covariance(self) -> np.ndarray:
        """Posterior covariance of every contrast, cell-major."""
        return np.linalg.inv(self.precision)

    def contrasts(self, cell: Hashable) -> Tuple[np.ndarray, np.ndarray]:
        """Posterior mean and covariance of one cell's ``K - 1`` contrasts."""
        c = self._cell_index[cell]
        block = slice(c * self._k, (c + 1) * self._k)
        return self.mu[block].copy(), self.covariance()[block, block]

    # ------------------------------------------------------------------ action
    def allocate(self, cells: Optional[Sequence[Hashable]] = None, draw: int = 100000,
                 aggressive: float = 1.0, floor: float = 0.0,
                 rng: Optional[np.random.Generator] = None) -> Dict[Hashable, Allocation]:
        """Each cell's next allocation, from one set of joint contrast draws.

        The arguments are ``LogisticBandit.allocate``'s.  Before the first
        informative batch every cell gets the uniform start-up allocation.
        """
        if draw <= 0:
            raise ValueError(f"draw must be positive, got {draw}")
        if aggressive <= 0:
            raise ValueError(f"aggressive must be positive, got {aggressive}")
        if not 0.0 <= floor < 1.0:
            raise ValueError(f"floor must be in [0, 1), got {floor}")
        cells = list(cells) if cells is not None else list(self.cells)
        K = len(self.arms)
        if not self.fitted:
            nan = float("nan")
            return {cell: Allocation(list(self.arms), {a: 1.0 / K for a in self.arms},
                                     {a: nan for a in self.arms}, {a: nan for a in self.arms})
                    for cell in cells}
        gen = rng if rng is not None else np.random.default_rng()
        draws = gen.multivariate_normal(self.mu, self.covariance(), size=draw, method="svd")
        out: Dict[Hashable, Allocation] = {}
        for cell in cells:
            c = self._cell_index[cell]
            scores = np.concatenate(
                (draws[:, c * self._k:(c + 1) * self._k], np.zeros((draw, 1))), axis=1)
            counts = np.bincount(scores.argmax(axis=1), minlength=K).astype(float)
            raw = counts / counts.sum()
            loss = (scores.max(axis=1, keepdims=True) - scores).mean(axis=0)
            shaped = counts ** aggressive
            p = shaped / shaped.sum()
            if floor > 0.0:
                p = _apply_floor(p, floor)
            out[cell] = Allocation(
                list(self.arms),
                {a: float(p[i]) for i, a in enumerate(self.arms)},
                {a: float(raw[i]) for i, a in enumerate(self.arms)},
                {a: float(loss[i]) for i, a in enumerate(self.arms)})
        return out
