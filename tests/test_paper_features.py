"""The parts of the 2026 paper that are code: Algorithm 1's memory rule, the
controls, the edge cases, the warm start, the stopping quantities, and the
diagnostics.  Each test names the paper section it checks."""

import warnings

import numpy as np
import pytest

from orts import LogisticBandit, TSPar, DiscountedTSPar, diagnostics


def _shift(obs, delta):
    """Move every arm's log odds by ``delta`` (a common level shift)."""
    out = {}
    for a, (n, c) in obs.items():
        p = c / n
        q = 1 / (1 + np.exp(-(np.log(p / (1 - p)) + delta)))
        out[a] = [n, int(round(n * q))]
    return out


def test_ranking_invariance_section_2_2():
    """Winner probabilities read the contrasts only: moving the level coordinate
    of the state, by any amount, leaves them exactly unchanged."""
    b = LogisticBandit()
    b.update({"A": [50000, 1500], "B": [50000, 1650], "C": [50000, 1400]})
    p1 = b.win_prop(draw=50000, rng=np.random.default_rng(1))
    b.mu[-1] += 3.0                                   # a common shift of every arm's log odds
    p2 = b.win_prop(draw=50000, rng=np.random.default_rng(1))
    assert p1 == p2


def test_memory_rule_r2_discards_the_level_section_2_1():
    """After a level shift between batches, OR-TS's contrast and level are the
    new batch's own; Full-TS's carried level pulls its fit toward the past."""
    base = {"A": [30000, 900], "B": [30000, 990]}
    orts_, full = LogisticBandit(), LogisticBandit()
    for _ in range(3):
        orts_.update(base)
        full.update(base, odds_ratios_only=False)
    shifted = _shift(base, -1.5)
    orts_.update(shifted)
    full.update(shifted, odds_ratios_only=False)
    true_contrast = np.log(990 / 29010) - np.log(900 / 29100)          # B against A
    assert abs(orts_.get_par(["B", "A"])[0][0] - true_contrast) < 0.03
    ref = orts_.reference
    new_level = np.log(shifted[ref][1] / (30000 - shifted[ref][1]))
    assert abs(orts_.mu[-1] - new_level) < 0.05
    assert abs(full.get_par([ref])[0][0] - new_level) > abs(orts_.mu[-1] - new_level)


def test_properness_skip_supplement_a():
    """A batch with no events, or no non-events, is skipped and leaves the state."""
    b = LogisticBandit()
    b.update({"A": [1000, 30], "B": [1000, 25]})
    mu, S = b.mu.copy(), b.sigma_inv.copy()
    assert b.update({"A": [500, 0], "B": [500, 0]}) is False
    assert b.update({"A": [5, 5], "B": [5, 5]}) is False
    assert np.allclose(mu, b.mu) and np.allclose(S, b.sigma_inv)
    assert b.update({"A": [1000, 31], "B": [1000, 24]}) is True


def test_single_arm_batch_leaves_contrasts_unchanged_section_5_3():
    b = LogisticBandit()
    b.update({"A": [20000, 600], "B": [20000, 660]})
    contrast_before = b.get_par(["A", "B"])[0][0]
    b.update({"B": [20000, 400]})           # only the reference arm is exposed
    contrast_after = b.get_par(["A", "B"])[0][0]
    assert abs(contrast_before - contrast_after) < 1e-6


def test_decay_weakens_the_carried_precision_section_2_3():
    obs = {"A": [20000, 600], "B": [20000, 660]}
    plain, decayed = LogisticBandit(), LogisticBandit()
    for _ in range(4):
        plain.update(obs)
        decayed.update(obs, decay=0.5)
    assert decayed.contrast_sd()["A"] > plain.contrast_sd()["A"]


def test_discounted_beta_ts_matches_tempering_section_2_3():
    d = DiscountedTSPar(0.1)
    d.update({"A": [100, 10], "B": [100, 5]})
    a1, b1 = d.alpha.copy(), d.beta.copy()
    d.update({"A": [100, 10], "B": [100, 5]})
    # Beta(a,b)^(1-lambda) = Beta(1 + (1-lambda)(a-1), 1 + (1-lambda)(b-1)), then add the batch
    assert np.allclose(d.alpha, 1 + 0.9 * (a1 - 1) + np.array([10, 5]))
    assert np.allclose(d.beta, 1 + 0.9 * (b1 - 1) + np.array([90, 95]))


def test_aggressiveness_and_floors_section_5_2():
    b = LogisticBandit()
    b.update({"A": [50000, 1500], "B": [50000, 1650], "C": [50000, 1400]})
    rng = lambda: np.random.default_rng(3)
    flat = b.win_prop(draw=50000, aggressive=1.0, rng=rng())
    sharp = b.win_prop(draw=50000, aggressive=2.0, rng=rng())
    soft = b.win_prop(draw=50000, aggressive=0.5, rng=rng())
    best = max(flat, key=flat.get)
    assert sharp[best] > flat[best] > soft[best]
    floored = b.win_prop(draw=50000, floor=0.2, rng=rng())
    assert min(floored.values()) >= 0.2 - 1e-9 and abs(sum(floored.values()) - 1) < 1e-9


def test_expected_loss_orders_like_winner_probability_section_6_1():
    b = LogisticBandit()
    b.update({"A": [50000, 1500], "B": [50000, 1650], "C": [50000, 1400]})
    rng = np.random.default_rng(5)
    p = b.win_prop(draw=50000, rng=rng)
    loss = b.expected_loss(draw=50000, rng=np.random.default_rng(5))
    assert max(p, key=p.get) == min(loss, key=loss.get)
    assert all(v >= 0 for v in loss.values())


def test_warm_start_from_beta_posteriors_supplement_h():
    post = {"A": (301, 29701), "B": (291, 29711), "C": (311, 29691)}
    w = LogisticBandit.from_beta_posteriors(post, reference="C")
    assert w.action_list == ["A", "B", "C"]
    expect = np.log(301 / 29701) - np.log(311 / 29691)
    assert abs(w.mu[0] - expect) < 1e-9
    cov = w.covariance()
    vC = 1 / 311 + 1 / 29691
    assert abs(cov[0, 1] - vC) < 1e-9                     # shared-reference covariance
    assert abs(cov[0, 0] - (1 / 301 + 1 / 29701 + vC)) < 1e-9
    assert w.update({"A": [30000, 300], "B": [30000, 290], "C": [30000, 310]}) is True


def test_reference_change_leaves_winner_probabilities_unchanged_supplement_b():
    b = LogisticBandit()
    b.update({"A": [50000, 1500], "B": [50000, 1650], "C": [50000, 1400]})
    p_before = b.win_prop(draw=50000, rng=np.random.default_rng(9))
    b.transform(["B", "C", "A"])
    p_after = b.win_prop(draw=50000, rng=np.random.default_rng(9))
    for a in p_before:
        assert abs(p_before[a] - p_after[a]) < 0.02


def test_new_arm_gets_uniform_share_supplement_b():
    b = LogisticBandit()
    b.update({"A": [50000, 1500], "B": [50000, 1650]})
    p = b.win_prop(["A", "B", "D"], draw=20000, rng=np.random.default_rng(2))
    assert abs(p["D"] - 1 / 3) < 1e-12 and abs(sum(p.values()) - 1) < 1e-9


def test_diagnostics_recover_a_constant_contrast_section_4_1():
    rng = np.random.default_rng(11)
    n, level, contrast = 40000, -3.0, 0.2
    alphas, avars, betas, bvars = [], [], [], []
    for t in range(60):
        a_t = level + rng.normal(0, 0.3)                    # the level wanders
        p_ref = 1 / (1 + np.exp(-a_t)); p_trt = 1 / (1 + np.exp(-(a_t + contrast)))
        obs = {"ctl": [n, rng.binomial(n, p_ref)], "trt": [n, rng.binomial(n, p_trt)]}
        (a, av), c = diagnostics.batch_contrasts(obs, "ctl")
        alphas.append(a); avars.append(av); betas.append(c["trt"][0]); bvars.append(c["trt"][1])
    R = diagnostics.level_contrast_ratio(alphas, avars, betas, bvars)
    assert R > 3 or R == float("inf")
    assert diagnostics.excess_sd(alphas, avars) > 0.2
    assert diagnostics.lag1_autocorrelation(betas) < 0.3


def test_implied_decay_is_zero_without_drift_supplement_h():
    b = LogisticBandit()
    b.update({"A": [50000, 1500], "B": [50000, 1650]})
    assert b.implied_decay(0.0) == 0.0
    lam = b.implied_decay(0.05)
    assert 0 < lam < 1
    assert diagnostics.implied_decay(0.05 ** 2, 0.05 ** 2) == pytest.approx(0.5)


def test_root_shims_still_import_with_a_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        import importlib
        mod = importlib.import_module("logisticbandit")
        assert hasattr(mod, "LogisticBandit")
    assert any(issubclass(x.category, DeprecationWarning) for x in w)
