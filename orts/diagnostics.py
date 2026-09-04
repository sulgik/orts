"""Running diagnostics for the state-separation assumption (paper, Section 5.3).

All functions work on aggregated counts only.  The core quantities are the
per-batch level ``alpha_t = logit(p_ref,t)`` and contrast
``beta_t = logit(p_i,t) - logit(p_ref,t)`` with their delta-method sampling
variances, and the *excess* variance of a series of such estimates: the
variance across batches minus the mean sampling variance, floored at zero,
which is the movement of the true quantity beyond sampling noise
(paper, Section 4.1).
"""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .utils import logit


def batch_contrasts(obs: Dict[str, Sequence[float]], reference: str
                    ) -> Tuple[Tuple[float, float], Dict[str, Tuple[float, float]]]:
    """One batch's level and contrasts with their sampling variances.

    Returns ``((alpha, var_alpha), {arm: (beta, var_beta)})``.  The sampling
    variance of a log odds is ``1/events + 1/non_events``; a contrast adds the
    two arms'.  Cells with zero events or zero non-events give ``nan``.
    """
    if reference not in obs:
        raise KeyError(f"reference '{reference}' not in obs")

    def cell(n, c):
        n, c = float(n), float(c)
        if c <= 0 or n - c <= 0:
            return float("nan"), float("nan")
        return logit(c / n), 1.0 / c + 1.0 / (n - c)

    a_est, a_var = cell(*obs[reference])
    contrasts = {}
    for arm, (n, c) in obs.items():
        if arm == reference:
            continue
        est, var = cell(n, c)
        contrasts[arm] = (est - a_est, var + a_var)
    return (a_est, a_var), contrasts


def excess_variance(estimates: Sequence[float], sampling_vars: Sequence[float]) -> float:
    """``max(0, var(estimates) - mean(sampling_vars))`` over usable batches."""
    e = np.asarray(estimates, dtype=float)
    v = np.asarray(sampling_vars, dtype=float)
    keep = np.isfinite(e) & np.isfinite(v)
    if keep.sum() < 2:
        return 0.0
    return float(max(0.0, np.var(e[keep], ddof=1) - np.mean(v[keep])))


def excess_sd(estimates: Sequence[float], sampling_vars: Sequence[float]) -> float:
    return float(np.sqrt(excess_variance(estimates, sampling_vars)))


def level_contrast_ratio(alpha: Sequence[float], alpha_var: Sequence[float],
                         beta: Sequence[float], beta_var: Sequence[float]) -> float:
    """``R = excess sd(alpha) / excess sd(beta)`` over a series of batches.

    ``R >> 1`` says the level moved and the contrast did not (OR-TS's bet
    holds, Beta-TS's fails); ``R < 1`` says the contrast moved as well.
    ``inf`` when the contrast moved no more than sampling noise.
    """
    a, b = excess_sd(alpha, alpha_var), excess_sd(beta, beta_var)
    return float("inf") if b == 0.0 else a / b


def lag1_autocorrelation(x: Sequence[float]) -> float:
    """Lag-one autocorrelation: near zero for a constant seen through noise,
    positive for a quantity that drifts and stays where it went."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return float("nan")
    d = x - x.mean()
    den = float(np.sum(d * d))
    return float("nan") if den == 0 else float(np.sum(d[:-1] * d[1:]) / den)


def sampling_band(sampling_vars: Sequence[float], z: float = 2.0) -> np.ndarray:
    """``+-z`` sampling-sd half-widths for plotting a contrast series against noise."""
    return z * np.sqrt(np.asarray(sampling_vars, dtype=float))


def implied_decay(excess_var_beta: float, posterior_var: float) -> float:
    """``lambda = tau^2 / (v + tau^2)`` (paper, Supplement H)."""
    tau2, v = float(excess_var_beta), float(posterior_var)
    if tau2 < 0 or v < 0:
        raise ValueError("variances must be non-negative")
    return 0.0 if tau2 == 0.0 else tau2 / (v + tau2)
