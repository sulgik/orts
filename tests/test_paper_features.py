"""The parts of the 2026 paper that are code: Algorithm 1's memory rule, the
controls, the edge cases, the warm start, the stopping quantities, and the
diagnostics.  Each test names the paper section it checks."""

import warnings

import numpy as np
import pytest

from orts import LogisticBandit, TSPar, DiscountedTSPar, diagnostics
from orts.priors import (
    augment_symmetric_contrast_state,
    symmetric_contrast_covariance,
    symmetric_contrast_precision,
)


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


def test_allocate_is_the_action_and_its_arm_set_is_free():
    """`allocate` names the arms that will be live; the state's arm set need not match."""
    b = LogisticBandit(reference="A")
    b.update({"A": [50000, 1500], "B": [50000, 1650], "C": [50000, 1400]})
    q = b.allocate(["C", "B", "D"], draw=30000, rng=np.random.default_rng(6))   # A dropped, D never seen
    assert q.arms == ["C", "B", "D"] and list(q.shares) == ["C", "B", "D"]
    assert abs(sum(q.shares.values()) - 1) < 1e-9
    assert q.shares["D"] == pytest.approx(1 / 3) and np.isnan(q.p_best["D"]) and np.isnan(q.expected_loss["D"])
    assert q.p_best["B"] > q.p_best["C"] and q.expected_loss["B"] < q.expected_loss["C"]
    assert q.leader == "B" and q["B"] == q.shares["B"]
    assert b.action_list == ["B", "C", "A"]                                    # the state is untouched
    assert b.win_prop(["C", "B", "D"], draw=30000, rng=np.random.default_rng(6)) == q.shares


def test_first_fit_check_and_start_up_allocation_algorithm_1():
    """Before any fit `allocate` returns the start-up allocation.  Under the default
    symmetric prior a separated first batch is fitted; under the explicit flat
    option it has no finite fit and raises, and once a state exists a later
    separated batch is fitted because the carried prior identifies it."""
    b = LogisticBandit(reference="A")
    q = b.allocate(["A", "B", "C"], draw=1000)
    assert q.shares == {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3} and all(np.isnan(v) for v in q.p_best.values())

    flat = LogisticBandit(reference="A", contrast_prior="flat")
    with pytest.raises(RuntimeError, match="flat contrast prior"):
        flat.update({"A": [1000, 30], "B": [1000, 0], "C": [1000, 25]})   # B separated
    assert flat.action_list == [] and not flat.fitted
    assert flat.update({"A": [1000, 30], "B": [1000, 20], "C": [1000, 25]}) is True
    assert flat.update({"A": [1000, 30], "B": [1000, 0], "C": [1000, 25]}) is True  # B now has a prior
    assert flat.contrasts()["B"][0] < flat.contrasts()["C"][0]

    assert b.update({"A": [1000, 30], "B": [1000, 0], "C": [1000, 25]}) is True
    assert b.contrasts()["B"][0] < b.contrasts()["C"][0]


def test_symmetric_proper_initialization_is_the_default_supplement_a():
    """S_0 = tau^-2 (I - 11'/K) with tau = sqrt(2) by default: every pairwise
    difference has prior variance 2 tau^2 (so prior sd 2) and every arm prior
    winner probability 1/K, and it fits a first batch with zero cells."""
    b = LogisticBandit()
    assert b.contrast_prior == "symmetric"
    assert np.isclose(b.arm_effect_prior_sd, np.sqrt(2.0))

    tau, K = 0.5, 4
    S0 = symmetric_contrast_precision(K, tau)
    Sigma0 = np.linalg.inv(S0)
    assert np.allclose(Sigma0, symmetric_contrast_covariance(K, tau))
    assert np.allclose(Sigma0, tau ** 2 * (np.eye(K - 1) + np.ones((K - 1, K - 1))))
    # reference-vs-arm and arm-vs-arm differences share the variance 2 tau^2
    assert np.isclose(Sigma0[0, 0], 2 * tau ** 2)
    assert np.isclose(Sigma0[0, 0] + Sigma0[1, 1] - 2 * Sigma0[0, 1], 2 * tau ** 2)
    # Algorithm 1's default tau gives every pairwise contrast prior sd 2
    assert np.isclose(symmetric_contrast_covariance(K, np.sqrt(2.0))[0, 0], 4.0)

    scaled = LogisticBandit(arm_effect_prior_sd=tau)
    assert scaled.update({"A": [200, 0], "B": [200, 5], "C": [200, 4], "D": [200, 6]}) is True
    p = scaled.win_prop(draw=20000, rng=np.random.default_rng(3))
    assert p["A"] < min(p["B"], p["C"], p["D"])


def test_symmetric_augmentation_of_a_new_arm_supplement_b():
    """The default augmentation leaves the incumbents' pairwise posteriors
    alone, gives the newcomer the retained latent centre plus tau^2, and
    shrinks a separated newcomer that the flat option leaves unidentified."""
    tau = 1.0
    mean = np.array([0.4, -0.2])
    cov = np.array([[0.5, 0.1], [0.1, 0.3]])
    aug_mean, aug_cov = augment_symmetric_contrast_state(mean, cov, 1, tau)
    assert np.allclose(aug_cov[:2, :2], cov)                      # incumbents untouched
    assert np.allclose(aug_mean[:2], mean)
    n_actions = 3                                                 # 2 contrasts + reference
    expected_var = tau ** 2 + tau ** 2 / n_actions + float(np.ones(2) @ cov @ np.ones(2)) / n_actions ** 2
    assert np.isclose(aug_cov[2, 2], expected_var)
    assert np.isclose(aug_mean[2], mean.sum() / n_actions)

    obs = {"A": [2000, 60], "B": [2000, 66]}
    default = LogisticBandit(reference="A")
    flat = LogisticBandit(reference="A", contrast_prior="flat")
    for b in (default, flat):
        b.update(obs)
    assert default.update({"A": [2000, 60], "B": [2000, 66], "D": [2000, 0]}) is True
    with pytest.raises(RuntimeError, match="flat contrast prior"):
        flat.update({"A": [2000, 60], "B": [2000, 66], "D": [2000, 0]})
    # the incumbent comparison survives the arrival
    assert default.contrasts()["B"][0] > 0
    assert np.isfinite(default.contrasts()["D"][0]) and default.contrasts()["D"][0] < 0


def test_deprecated_prior_spellings_still_work():
    """2.2's init_scale/new_arm_scale map onto the named priors, with a warning."""
    with pytest.warns(DeprecationWarning):
        b = LogisticBandit(init_scale=0.5)
    assert b.contrast_prior == "symmetric" and b.arm_effect_prior_sd == 0.5
    with pytest.warns(DeprecationWarning):
        b = LogisticBandit(new_arm_scale=1.0)
    assert b.contrast_prior == "independent" and b.new_contrast_prior_sd == 1.0


def test_disconnected_batch_starts_its_own_group_supplement_b():
    b = LogisticBandit()
    b.update({"A": [10000, 300], "B": [10000, 330]})
    assert b.update({"C": [10000, 350], "D": [10000, 300]}) is True
    assert [sorted(g) for g in b.groups()] == [["A", "B"], ["C", "D"]]
    # each group is a posterior of its own; neither knows the other's arms
    assert abs(sum(b.allocate(["A", "B"], draw=20000, rng=np.random.default_rng(0)
                           ).shares.values()) - 1) < 1e-9
    # one call may span both: Supplement B's new-arm rule read for a group, so
    # each group takes traffic in proportion to its size and allocates inside
    # itself.  No comparison between the groups is invented, so no ranking is
    # reported across them.
    q = b.allocate(["A", "B", "C", "D"], draw=20000, rng=np.random.default_rng(0))
    assert abs(sum(q.shares.values()) - 1) < 1e-9
    assert abs(q.shares["A"] + q.shares["B"] - 0.5) < 1e-9      # group {A, B}
    assert abs(q.shares["C"] + q.shares["D"] - 0.5) < 1e-9      # group {C, D}
    assert q.shares["B"] > q.shares["A"] and q.shares["C"] > q.shares["D"]
    assert all(np.isnan(v) for v in q.p_best.values())
    assert all(np.isnan(v) for v in q.expected_loss.values())
    # within one group the ranking is reported as usual
    assert not np.isnan(b.allocate(["A", "B"], draw=20000,
                                rng=np.random.default_rng(0)).p_best["B"])


def test_group_rule_reduces_to_the_paper_s_new_arm_rule_supplement_b():
    """A group of one arm with no posterior is the rule the paper states: the
    uniform share 1/|A|, with the observed arms scaled into the remainder."""
    b = LogisticBandit()
    b.update({"A": [50000, 1500], "B": [50000, 1650]})
    q = b.allocate(["A", "B", "E"], draw=20000, rng=np.random.default_rng(2))
    assert abs(q.shares["E"] - 1 / 3) < 1e-12
    assert abs(q.shares["A"] + q.shares["B"] - 2 / 3) < 1e-9
    assert np.isnan(q.p_best["E"]) and not np.isnan(q.p_best["B"])   # one group: ranked


def test_the_bridge_keeps_one_group_supplement_b():
    """The paper's bridge: batch one compares A with B, batch two B with C.
    The shared arm keeps this inside one group -- C joins by augmentation --
    and the joint posterior carries an A-versus-C comparison never run
    directly, with the wider uncertainty the indirect route implies."""
    b = LogisticBandit(reference="A")
    b.update({"A": [20000, 600], "B": [20000, 660]})
    b.update({"B": [20000, 660], "C": [20000, 700]})
    assert len(b.groups()) == 1 and sorted(b.groups()[0]) == ["A", "B", "C"]
    # marginals come from the covariance, not from a precision sub-block
    cov = b.covariance()
    i, j = b.action_list.index("B"), b.action_list.index("C")
    sd_ab = float(np.sqrt(cov[i, i]))                          # run directly, batch one
    sd_bc = float(np.sqrt(cov[i, i] + cov[j, j] - 2 * cov[i, j]))   # batch two
    sd_ac = float(np.sqrt(cov[j, j]))                          # never run directly
    # A was not in batch two, so its comparison with B is what batch one left it
    assert abs(sd_ab - 0.05729) < 1e-4
    # the indirect route carries both batches' uncertainty
    assert sd_ac > sd_ab and sd_ac > sd_bc
    assert abs(sd_ac - np.hypot(sd_ab, sd_bc)) < 1e-4
    q = b.allocate(["A", "B", "C"], draw=20000, rng=np.random.default_rng(0))
    assert not np.isnan(q.p_best["C"]) and abs(sum(q.shares.values()) - 1) < 1e-9


def test_a_batch_joining_two_separate_groups_is_refused_supplement_b():
    """Two groups initialized independently carry contrast priors centred on
    their own arm sets.  Joining them is not a construction the paper gives,
    so the batch is refused rather than fitted on an invented one."""
    b = LogisticBandit()
    b.update({"A": [10000, 300], "B": [10000, 330]})
    b.update({"C": [10000, 350], "D": [10000, 300]})
    assert len(b.groups()) == 2
    with pytest.raises(ValueError, match="separate groups"):
        b.update({"B": [10000, 330], "C": [10000, 350]})
    # the refusal leaves both states exactly as they were
    assert [sorted(g) for g in b.groups()] == [["A", "B"], ["C", "D"]]


def test_query_still_works_as_a_deprecated_alias_of_allocate():
    """`query` was the 2.1-2.2 spelling; it forwards, with a warning."""
    b = LogisticBandit()
    b.update({"A": [50000, 1500], "B": [50000, 1650]})
    with pytest.warns(DeprecationWarning, match="query is deprecated"):
        old = b.query(["A", "B"], draw=20000, rng=np.random.default_rng(4))
    new = b.allocate(["A", "B"], draw=20000, rng=np.random.default_rng(4))
    assert old.shares == new.shares and old.p_best == new.p_best
    assert old.leader == new.leader
