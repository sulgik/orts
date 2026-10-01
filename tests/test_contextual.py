"""ContextualLogisticBandit: its limits are the models already in the package."""

import math

import numpy as np
import pytest

from orts import ContextualLogisticBandit, LogisticBandit

ARMS = ["A", "B", "C"]
BATCHES = [
    {"x": {"A": [3000, 95], "B": [2500, 90], "C": [3500, 100]},
     "y": {"A": [900, 20], "B": [1100, 31], "C": [1000, 30]}},
    {"x": {"A": [1000, 40], "B": [4000, 170], "C": [2000, 70]},
     "y": {"A": [1500, 33], "B": [500, 16], "C": [1000, 27]}},
]


def fit(arms=ARMS, cells=("x", "y"), batches=BATCHES, **kwargs):
    kwargs.setdefault("interaction_sd", 0.25)
    bandit = ContextualLogisticBandit(arms, list(cells), **kwargs)
    for batch in batches:
        bandit.update({c: batch[c] for c in cells})
    return bandit


def test_one_cell_is_logistic_bandit():
    # tau^2 + omega^2 = 2 is LogisticBandit's default arm-effect variance
    ctx = fit(cells=("x",), arm_effect_prior_sd=1.0, interaction_sd=1.0)
    ref = LogisticBandit(reference="C", arm_effect_prior_sd=math.sqrt(2.0))
    for batch in BATCHES:
        ref.update(batch["x"])
    mean, cov = ctx.contrasts("x")
    np.testing.assert_allclose(mean, ref.mu[:2], atol=1e-4)
    np.testing.assert_allclose(cov, ref.covariance()[:2, :2], atol=1e-5)


def test_large_interaction_sd_decouples_the_cells():
    ctx = fit(arm_effect_prior_sd=1e-3, interaction_sd=math.sqrt(2.0))
    for cell in ("x", "y"):
        ref = LogisticBandit(reference="C", arm_effect_prior_sd=math.sqrt(2.0))
        for batch in BATCHES:
            ref.update(batch[cell])
        mean, cov = ctx.contrasts(cell)
        np.testing.assert_allclose(mean, ref.mu[:2], atol=1e-3)
        np.testing.assert_allclose(cov, ref.covariance()[:2, :2], atol=1e-4)


def test_small_interaction_sd_gives_every_cell_the_same_contrasts():
    ctx = fit(interaction_sd=1e-4)
    mean_x, cov_x = ctx.contrasts("x")
    mean_y, cov_y = ctx.contrasts("y")
    np.testing.assert_allclose(mean_x, mean_y, atol=1e-4)
    np.testing.assert_allclose(cov_x, cov_y, atol=1e-5)


def test_thin_cell_borrows_from_the_others():
    # "y" has about a third of "x"'s data, so pooling tightens it the most
    loose = fit(interaction_sd=5.0)
    tight = fit(interaction_sd=0.1)
    ratio = {cell: np.diag(tight.contrasts(cell)[1]) / np.diag(loose.contrasts(cell)[1])
             for cell in ("x", "y")}
    assert np.all(ratio["x"] < 1.0)
    assert np.all(ratio["y"] < ratio["x"])
    assert np.all(ratio["y"] < 0.7)


def test_cell_level_shift_leaves_the_contrasts_alone():
    # Doubling every arm's odds in one cell moves that cell's intercept only.
    # With a prior this wide the posterior mean is the maximum likelihood fit,
    # so the contrasts must come out the same.
    def shifted(arm_counts, factor):
        out = {}
        for arm, (n, s) in arm_counts.items():
            odds = factor * s / (n - s)
            out[arm] = [n, n * odds / (1.0 + odds)]
        return out

    wide = dict(arm_effect_prior_sd=100.0, interaction_sd=100.0)
    base = fit(batches=BATCHES[:1], **wide)
    moved = ContextualLogisticBandit(ARMS, ["x", "y"], **wide)
    moved.update({"x": shifted(BATCHES[0]["x"], 2.0), "y": BATCHES[0]["y"]})
    np.testing.assert_allclose(moved.mu, base.mu, atol=1e-4)
    assert np.all(np.diag(moved.contrasts("x")[1]) < np.diag(base.contrasts("x")[1]))


def test_pairwise_posterior_does_not_depend_on_the_reference():
    first = fit()
    reordered = fit(arms=["C", "B", "A"])
    for cell in ("x", "y"):
        mean, cov = first.contrasts(cell)            # A - C, B - C
        mean_r, cov_r = reordered.contrasts(cell)    # C - A, B - A
        np.testing.assert_allclose(mean[0], -mean_r[0], atol=1e-5)
        np.testing.assert_allclose(mean[1] - mean[0], mean_r[1], atol=1e-5)
        np.testing.assert_allclose(cov[0, 0], cov_r[0, 0], atol=1e-6)
        np.testing.assert_allclose(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1], cov_r[1, 1], atol=1e-6)


def simulate(contrast_by_cell, periods=10, n=20000, seed=0):
    """Feed an "auto" bandit balanced batches from cells with the given log-odds contrasts."""
    rng = np.random.default_rng(seed)
    cells = list(range(len(contrast_by_cell)))
    bandit = ContextualLogisticBandit(ARMS, cells)
    for _ in range(periods):
        batch = {}
        for c, contrast in enumerate(contrast_by_cell):
            p = 1.0 / (1.0 + np.exp(-(-3.0 + rng.normal(0.0, 0.3) + np.asarray(contrast))))
            batch[c] = {a: [n, rng.binomial(n, p[i])] for i, a in enumerate(ARMS)}
        bandit.update(batch)
    return bandit


def test_auto_interaction_sd_pools_cells_that_agree():
    assert ContextualLogisticBandit(ARMS, ["x"]).estimates_interaction_sd
    bandit = simulate([[0.2, 0.1, 0.0]] * 6)
    assert bandit.interaction_sd <= 0.05
    spread = np.ptp([bandit.contrasts(c)[0] for c in range(6)], axis=0)
    assert np.all(spread < 0.05)


def test_auto_interaction_sd_separates_cells_that_differ():
    rng = np.random.default_rng(1)
    truth = [[0.2 + rng.normal(0, 0.4), 0.1 + rng.normal(0, 0.4), 0.0] for _ in range(6)]
    bandit = simulate(truth)
    assert bandit.interaction_sd >= 0.15
    for c in range(6):
        np.testing.assert_allclose(bandit.contrasts(c)[0], truth[c][:2], atol=0.12)


def test_auto_matches_a_fixed_fit_at_the_chosen_value():
    # the posterior re-formed under the estimate is the fixed-prior posterior, up
    # to where each batch's Laplace approximation was expanded
    auto = ContextualLogisticBandit(ARMS, ["x", "y"])
    for batch in BATCHES:
        auto.update(batch)
    fixed = fit(interaction_sd=auto.interaction_sd)
    np.testing.assert_allclose(auto.mu, fixed.mu, atol=2e-3)
    np.testing.assert_allclose(auto.covariance(), fixed.covariance(), rtol=2e-2, atol=1e-5)


def test_uninformative_cells_are_skipped():
    bandit = ContextualLogisticBandit(ARMS, ["x", "y"])
    assert not bandit.update({"x": {"A": [50, 0], "B": [50, 0], "C": [50, 0]}})
    assert not bandit.fitted
    assert not bandit.update({"x": {"A": [50, 3]}})  # one arm compares nothing
    assert bandit.update({"x": {"A": [500, 30], "B": [500, 20]},  # reference arm absent
                          "y": {"A": [40, 0], "C": [40, 0]}})
    assert bandit.fitted
    assert np.all(np.isfinite(bandit.mu))


def test_allocation():
    bandit = ContextualLogisticBandit(ARMS, ["x", "y"])
    start = bandit.allocate()
    assert start["x"].shares == {a: pytest.approx(1 / 3) for a in ARMS}
    bandit = fit()
    out = bandit.allocate(draw=20000, floor=0.05, rng=np.random.default_rng(0))
    assert set(out) == {"x", "y"}
    for allocation in out.values():
        assert sum(allocation.shares.values()) == pytest.approx(1.0)
        assert min(allocation.shares.values()) >= 0.05 - 1e-12
        assert sum(allocation.p_best.values()) == pytest.approx(1.0)
    assert out["x"].leader == "B"
    assert list(bandit.allocate(["y"], draw=1000, rng=np.random.default_rng(0))) == ["y"]


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        ContextualLogisticBandit(["A"], ["x"])
    with pytest.raises(ValueError):
        ContextualLogisticBandit(ARMS, ["x"], interaction_sd=0.0)
    with pytest.raises(ValueError):
        ContextualLogisticBandit(ARMS, ["x"], interaction_sd="learned")
    bandit = ContextualLogisticBandit(ARMS, ["x"])
    with pytest.raises(KeyError):
        bandit.update({"z": {"A": [10, 1], "B": [10, 2]}})
    with pytest.raises(KeyError):
        bandit.update({"x": {"A": [10, 1], "D": [10, 2]}})
    with pytest.raises(ValueError):
        bandit.update({"x": {"A": [10, 11], "B": [10, 2]}})
