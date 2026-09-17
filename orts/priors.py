"""The contrast prior: symmetric and proper, defined through latent arm effects.

Algorithm 1 starts from exchangeable arm effects ``z_i ~ iid N(0, tau^2)``.
With arm ``K`` as the reference the stored coordinates are the contrasts
``beta_i = z_i - z_K``, whose prior is

    mu_0 = 0,   Sigma_0 = tau^2 (I + 11'),   S_0 = tau^-2 (I - 11'/K).

Every pairwise difference then has variance ``2 tau^2`` and every arm prior
winner probability ``1/K``, and neither depends on which arm was made the
reference (paper, Supplement A).  Independent equal-variance priors on the
reference contrasts do not have that symmetry: they would give a
reference-versus-arm difference a different variance from a difference
between two non-reference arms.

When an arm joins later it enters through ``augment_symmetric_contrast_state``,
the augmentation step of Supplement B's operational sequence, which leaves
the incumbents' pairwise posteriors and the reference invariance of the state
untouched.
"""

from typing import Tuple

import numpy as np


def _validated_action_count(n_actions: int) -> int:
    if isinstance(n_actions, (bool, np.bool_)) or not isinstance(n_actions, (int, np.integer)):
        raise TypeError("n_actions must be an integer")
    n_actions = int(n_actions)
    if n_actions < 2:
        raise ValueError("n_actions must be at least 2")
    return n_actions


def _validated_arm_effect_sd(arm_effect_sd: float) -> float:
    if isinstance(arm_effect_sd, (bool, np.bool_)):
        raise TypeError("arm_effect_sd must be a positive finite scalar")
    try:
        arm_effect_sd = float(arm_effect_sd)
    except (TypeError, ValueError) as exc:
        raise TypeError("arm_effect_sd must be a positive finite scalar") from exc
    if not np.isfinite(arm_effect_sd) or arm_effect_sd <= 0.0:
        raise ValueError("arm_effect_sd must be positive and finite")
    return arm_effect_sd


def symmetric_contrast_covariance(n_actions: int, arm_effect_sd: float) -> np.ndarray:
    """``Sigma_0 = tau^2 (I + 11')`` over the ``K - 1`` reference contrasts."""
    n_actions = _validated_action_count(n_actions)
    tau = _validated_arm_effect_sd(arm_effect_sd)
    n_contrasts = n_actions - 1
    return tau ** 2 * (np.eye(n_contrasts) + np.ones((n_contrasts, n_contrasts)))


def symmetric_contrast_precision(n_actions: int, arm_effect_sd: float) -> np.ndarray:
    """``S_0 = tau^-2 (I - 11'/K)``, the inverse of the covariance above."""
    n_actions = _validated_action_count(n_actions)
    tau = _validated_arm_effect_sd(arm_effect_sd)
    n_contrasts = n_actions - 1
    return (np.eye(n_contrasts) - np.ones((n_contrasts, n_contrasts)) / n_actions) / tau ** 2


def augment_symmetric_contrast_state(mean, covariance, n_new: int,
                                     arm_effect_sd: float) -> Tuple[np.ndarray, np.ndarray]:
    """Add ``n_new`` arms to a contrast state without disturbing the incumbents.

    ``mean`` and ``covariance`` describe ``K - 1`` contrasts against one
    existing reference arm.  The centre of the ``K`` latent arm effects, which
    the contrast state does not identify, is retained as an independent
    ``N(0, tau^2 / K)``; each new latent effect is then drawn from the same
    ``N(0, tau^2)`` population as the incumbents' and contrasted against the
    same reference arm.

    The existing contrasts come first in the result and their block is
    returned unchanged, so their pairwise posteriors are preserved.  The
    appended coordinates share the uncertainty in that retained centre, which
    is the off-diagonal term they carry (paper, Supplement B).
    """
    if isinstance(n_new, (bool, np.bool_)) or not isinstance(n_new, (int, np.integer)):
        raise TypeError("n_new must be an integer")
    n_new = int(n_new)
    if n_new < 1:
        raise ValueError("n_new must be at least 1")
    tau = _validated_arm_effect_sd(arm_effect_sd)

    try:
        mean = np.asarray(mean, dtype=float)
        covariance = np.asarray(covariance, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("mean and covariance must be numeric array-like values") from exc

    if mean.ndim != 1:
        raise ValueError("mean must be one-dimensional")
    n_contrasts = mean.size
    if covariance.shape != (n_contrasts, n_contrasts):
        raise ValueError("covariance shape must be (len(mean), len(mean))")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(covariance)):
        raise ValueError("mean and covariance must contain only finite values")
    if not np.allclose(covariance, covariance.T, rtol=1e-12, atol=1e-12):
        raise ValueError("covariance must be symmetric")

    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues = np.linalg.eigvalsh(covariance)
    tolerance = 1e-12 * max(1.0, float(np.max(np.abs(eigenvalues), initial=0.0)))
    if eigenvalues.size and eigenvalues[0] < -tolerance:
        raise ValueError("covariance must be positive semidefinite")

    n_actions = n_contrasts + 1
    ones = np.ones(n_contrasts)
    reference_loading = covariance @ ones / n_actions
    reference_variance = (tau ** 2 / n_actions
                          + float(ones @ covariance @ ones) / n_actions ** 2)

    augmented_mean = np.concatenate((
        mean.copy(),
        np.full(n_new, float(np.sum(mean)) / n_actions),
    ))
    augmented_covariance = np.empty((n_contrasts + n_new, n_contrasts + n_new), dtype=float)
    augmented_covariance[:n_contrasts, :n_contrasts] = covariance
    augmented_covariance[:n_contrasts, n_contrasts:] = reference_loading[:, None]
    augmented_covariance[n_contrasts:, :n_contrasts] = reference_loading[None, :]
    augmented_covariance[n_contrasts:, n_contrasts:] = (
        reference_variance * np.ones((n_new, n_new)) + tau ** 2 * np.eye(n_new))
    return augmented_mean, augmented_covariance
