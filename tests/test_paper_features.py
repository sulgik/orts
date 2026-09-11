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
    assert decayed.contrast_sd()["B"] > plain.contrast_sd()["B"]


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


def test_canonical_order_is_stable_under_batch_order():
    """The state's order is first-seen with the reference last, whatever order
    a batch names the arms in, and the same counts give the same allocation."""
    obs = {"A": [40000, 1200], "B": [40000, 1320], "C": [40000, 1160]}
    b1, b2 = LogisticBandit(), LogisticBandit()
    b1.update(obs); b1.update(obs)
    b2.update(obs); b2.update({"C": obs["C"], "A": obs["A"], "B": obs["B"]})
    assert b1.action_list == ["B", "C", "A"] == b2.action_list        # A: first arm seen, so the reference
    assert np.allclose(b1.mu, b2.mu) and np.allclose(b1.sigma_inv, b2.sigma_inv)
    p1 = b1.win_prop(["C", "A", "B"], draw=20000, rng=np.random.default_rng(4))
    assert list(p1) == ["C", "A", "B"]                                  # the caller's order


def test_reference_can_be_chosen_and_unobserved_batches_map_back():
    b = LogisticBandit(reference="control")
    b.update({"v1": [30000, 930], "control": [30000, 900], "v2": [30000, 990]})
    assert b.reference == "control" and b.action_list == ["v1", "v2", "control"]
    before = b.contrasts()
    b.update({"v1": [30000, 940], "v2": [30000, 985]})               # the reference is not exposed
    assert b.action_list == ["v1", "v2", "control"]                   # order unchanged
    after = b.contrasts()
    assert abs(after["v2"][0] - before["v2"][0]) < 0.1                # contrasts stayed coherent
    assert after["v1"][1] < before["v1"][1]                           # and v1's contrast sharpened


def test_drop_and_set_reference():
    b = LogisticBandit()
    b.update({"A": [30000, 900], "B": [30000, 990], "C": [30000, 960]})
    p_before = b.win_prop(["B", "C"], draw=30000, rng=np.random.default_rng(8))
    b.drop(["A"])
    assert b.action_list == ["B", "C"] or b.action_list == ["C", "B"]
    p_after = b.win_prop(["B", "C"], draw=30000, rng=np.random.default_rng(8))
    for a in p_before:
        assert abs(p_before[a] - p_after[a]) < 0.03
    b.set_reference("B")
    assert b.reference == "B"


def test_query_is_the_action_and_its_arm_set_is_free():
    """A query names the arms that will be live; the state's arm set need not match."""
    b = LogisticBandit(reference="A")
    b.update({"A": [50000, 1500], "B": [50000, 1650], "C": [50000, 1400]})
    q = b.query(["C", "B", "D"], draw=30000, rng=np.random.default_rng(6))   # A dropped, D never seen
    assert q.arms == ["C", "B", "D"] and list(q.shares) == ["C", "B", "D"]
    assert abs(sum(q.shares.values()) - 1) < 1e-9
    assert q.shares["D"] == pytest.approx(1 / 3) and np.isnan(q.p_best["D"]) and np.isnan(q.expected_loss["D"])
    assert q.p_best["B"] > q.p_best["C"] and q.expected_loss["B"] < q.expected_loss["C"]
    assert q.leader == "B" and q["B"] == q.shares["B"]
    assert b.action_list == ["B", "C", "A"]                                    # the state is untouched
    assert b.win_prop(["C", "B", "D"], draw=30000, rng=np.random.default_rng(6)) == q.shares


def test_first_fit_check_and_start_up_allocation_algorithm_1():
    """Before any fit the query is the start-up allocation; a first batch with
    a separated arm is not fitted under the flat prior; a later batch with a
    separated arm is fitted, because the carried prior identifies it."""
    b = LogisticBandit(reference="A")
    q = b.query(["A", "B", "C"], draw=1000)
    assert q.shares == {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3} and all(np.isnan(v) for v in q.p_best.values())
    assert b.update({"A": [1000, 30], "B": [1000, 0], "C": [1000, 25]}) is False    # B separated: no state formed
    assert b.action_list == [] and not b.fitted
    assert b.update({"A": [1000, 30], "B": [1000, 20], "C": [1000, 25]}) is True
    assert b.update({"A": [1000, 30], "B": [1000, 0], "C": [1000, 25]}) is True     # now B is identified by its prior
    assert b.contrasts()["B"][0] < b.contrasts()["C"][0]


def test_symmetric_proper_initialization_supplement_a():
    """S_0 = tau^-2 (I - 11'/K): every pairwise difference has prior variance
    2 tau^2 and every arm prior winner probability 1/K; it fits a first batch
    with zero cells."""
    tau, K = 0.5, 4
    b = LogisticBandit(init_scale=tau)
    S0 = (np.eye(K - 1) - np.ones((K - 1, K - 1)) / K) / tau ** 2
    Sigma0 = np.linalg.inv(S0)
    assert np.allclose(Sigma0, tau ** 2 * (np.eye(K - 1) + np.ones((K - 1, K - 1))))
    # reference-vs-arm and arm-vs-arm differences share the variance 2 tau^2
    assert np.isclose(Sigma0[0, 0], 2 * tau ** 2)
    assert np.isclose(Sigma0[0, 0] + Sigma0[1, 1] - 2 * Sigma0[0, 1], 2 * tau ** 2)
    assert b.update({"A": [200, 0], "B": [200, 5], "C": [200, 4], "D": [200, 6]}) is True
    p = b.win_prop(draw=20000, rng=np.random.default_rng(3))
    assert p["A"] < min(p["B"], p["C"], p["D"])


def test_proper_prior_for_a_separated_new_arm_supplement_c():
    """A new arm whose first batch is separated: flat by default, so its
    contrast is only as identified as the batch allows; with new_arm_scale
    it gets a proper N(0, s^2) prior and a finite, shrunk contrast."""
    obs = {"A": [2000, 60], "B": [2000, 66]}
    flat, proper = LogisticBandit(reference="A"), LogisticBandit(reference="A", new_arm_scale=1.0)
    for b in (flat, proper):
        b.update(obs)
        assert b.update({"A": [2000, 60], "B": [2000, 66], "D": [2000, 0]}) is True
    assert abs(proper.contrasts()["D"][0]) < abs(flat.contrasts()["D"][0])
    assert proper.contrasts()["D"][1] < flat.contrasts()["D"][1]
