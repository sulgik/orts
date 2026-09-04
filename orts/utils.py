"""Numerical core: the per-batch Laplace fit of the reference-coded logistic model.

The state passed around here is the pair ``(mu, sigma_inv)`` over the
coordinates ``(beta_1, ..., beta_{K-1}, alpha)``: the contrasts of every
non-reference arm against the reference, followed by the reference arm's own
log odds, which is the batch level.  ``estimate`` fits one batch: it maximises
the log posterior under a Gaussian prior on the carried coordinates (scaled by
``1 - discount``) and a flat prior on the new intercept, then returns the mode
and the negative Hessian at the mode.  Supplement A of the paper derives both.
"""

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

Prior = Tuple[Optional[np.ndarray], Optional[np.ndarray]]


def logistic(x: float, trunc: float = 9.0) -> float:
    """Numerically safe inverse logit of a scalar."""
    x = float(np.clip(x, -trunc, trunc))
    if x > 0:
        return 1.0 / (1.0 + np.exp(-x))
    return np.exp(x) / (1.0 + np.exp(x))


def logit(p: float, eps: float = 1e-3) -> float:
    """Log odds of a probability, clipped away from 0 and 1."""
    p = float(np.clip(p, eps, 1.0 - eps))
    return float(np.log(p / (1.0 - p)))


def is_pos_semidef(x: np.ndarray, tol: float = 0.0) -> bool:
    """True when every eigenvalue of the symmetric matrix ``x`` is >= -tol."""
    return bool(np.all(np.linalg.eigvalsh(np.asarray(x, dtype=float)) >= -tol))


def estimate(prior: Prior, obs: Sequence[Sequence[float]], indexes: Sequence[int],
             discount: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Fit one batch and return the Laplace state ``(mu_hat, sigma_inv)``.

    Parameters
    ----------
    prior
        ``(mu, sigma_inv)`` of the carried coordinates, or ``(None, None)``
        for the first batch.  Its last coordinate is the level; OR-TS passes
        it with zero precision (flat), Full-TS with the carried precision.
    obs
        ``[[exposures, events], ...]`` for the observed arms, new arms first,
        the reference arm last.
    indexes
        ``[n_new, n_new + n_unobserved, K]``: the layout of the parameter
        vector as new arms, carried-but-unobserved arms, observed arms.
    discount
        The decay ``lambda``; the prior precision enters scaled by ``1 - lambda``.
    """
    target_fn, gradient_fn = _build_fns(prior, obs, indexes, discount)
    initial_mu = _compute_initial(prior, obs, indexes)
    result = minimize(target_fn, initial_mu, jac=gradient_fn, method="Newton-CG")
    mu_hat = result.x
    sigma_inv = _estimate_sigma_inv(mu_hat, prior, obs, indexes, discount)
    return mu_hat, sigma_inv


def _compute_initial(prior: Prior, obs, indexes) -> np.ndarray:
    index_obs = list(range(indexes[0])) + list(range(indexes[1], indexes[2]))
    initial = np.zeros(indexes[2])
    p_ref = np.clip(float(obs[-1][1]) / float(obs[-1][0]), 1e-3, 1 - 1e-3)
    for i, j in enumerate(index_obs[:-1]):
        p = np.clip(float(obs[i][1]) / float(obs[i][0]), 1e-3, 1 - 1e-3)
        initial[j] = np.log(p / (1.0 - p)) - np.log(p_ref / (1.0 - p_ref))
    if prior[0] is not None:
        initial[indexes[0]:] = prior[0]
    return initial


def _estimate_sigma_inv(mu_hat, prior: Prior, obs, indexes, discount) -> np.ndarray:
    """Negative Hessian at the mode: discounted prior precision plus the
    binomial information ``X^T W X`` of Supplement A."""
    K = indexes[2]
    sigma_inv_prior = np.zeros((K, K))
    if prior[1] is not None:
        sigma_inv_prior[indexes[0]:, indexes[0]:] = prior[1]

    index_obs = list(range(indexes[0])) + list(range(indexes[1], indexes[2]))
    p = np.array([logistic(mu + mu_hat[-1]) for mu in mu_hat[index_obs][:-1]])
    p_ref = logistic(mu_hat[-1])
    total_cnt = np.array([float(i[0]) for i in obs[:-1]])
    total_cnt_ref = float(obs[-1][0])
    npq = total_cnt * p * (1.0 - p)
    npq_ref = total_cnt_ref * p_ref * (1.0 - p_ref)

    info = np.zeros((K, K))
    for i, j in enumerate(index_obs[:-1]):
        info[j, j] = npq[i]
        info[j, -1] = npq[i]
        info[-1, j] = npq[i]
    info[-1, -1] = np.sum(npq) + npq_ref
    return (1.0 - discount) * sigma_inv_prior + info


def _build_fns(prior: Prior, obs, indexes, discount) -> Tuple[Callable, Callable]:
    total_cnt = np.array([float(i[0]) for i in obs])
    success_cnt = np.array([float(i[1]) for i in obs])
    total_number = float(np.sum(total_cnt))
    index_prior = list(range(indexes[0], indexes[2]))
    index_obs = list(range(indexes[0])) + list(range(indexes[1], indexes[2]))
    has_prior = prior[0] is not None and len(index_prior) > 0

    def negative_log_posterior(w: np.ndarray) -> float:
        prior_log = 0.0
        if has_prior:
            d = np.asarray(w[index_prior]) - np.asarray(prior[0])
            prior_log = 0.5 * d.dot(prior[1]).dot(d) / total_number
        p = np.array([logistic(w[i] + w[-1]) for i in index_obs[:-1]])
        p = np.append(p, logistic(w[-1]))
        p = np.clip(p, 1e-3, 1 - 1e-3)
        nll = -np.sum(success_cnt * np.log(p) + (total_cnt - success_cnt) * np.log(1.0 - p)) / total_number
        return (1.0 - discount) * prior_log + nll

    def gradient(w: np.ndarray) -> np.ndarray:
        p = np.array([logistic(w[i] + w[-1]) for i in index_obs[:-1]])
        p = np.append(p, logistic(w[-1]))
        grad_prior = np.zeros(indexes[2])
        if has_prior:
            grad_prior[index_prior] = (np.asarray(w[index_prior]) - prior[0]).dot(prior[1]) / total_number
        grad_lik = np.zeros(indexes[2])
        grad_lik[index_obs] = -(success_cnt * (1.0 - p) - (total_cnt - success_cnt) * p) / total_number
        grad_lik[-1] = np.sum(grad_lik)
        return (1.0 - discount) * grad_prior + grad_lik

    return negative_log_posterior, gradient
