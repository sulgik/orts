"""
Unit tests for LinearBandit class.

Tests cover initialization, updates, input validation, Thompson sampling,
and edge cases for Gaussian (linear) bandits.
"""

import pytest
import numpy as np
from linearbandit import LinearBandit


class TestLinearBanditInitialization:
    """Test LinearBandit initialization and default values."""

    def test_default_initialization(self):
        """Test that LinearBandit initializes with empty defaults."""
        bandit = LinearBandit()
        assert bandit.mu == {}
        assert bandit.sigma == {}
        assert bandit.counts == {}
        assert bandit.sum_rewards == {}
        assert bandit.default_sigma == 1.0
        assert bandit.obs_noise == 1.0

    def test_initialization_with_priors(self):
        """Test initialization with prior parameters."""
        mu_prior = {"arm_1": 1.5, "arm_2": 2.0}
        sigma_prior = {"arm_1": 0.5, "arm_2": 0.3}

        bandit = LinearBandit(mu=mu_prior, sigma=sigma_prior)

        assert bandit.mu == mu_prior
        assert bandit.sigma == sigma_prior
        assert "arm_1" in bandit.counts
        assert "arm_2" in bandit.counts

    def test_initialization_with_custom_noise(self):
        """Test initialization with custom observation noise."""
        bandit = LinearBandit(obs_noise=2.0, default_sigma=3.0)
        assert bandit.obs_noise == 2.0
        assert bandit.default_sigma == 3.0

    def test_invalid_default_sigma(self):
        """Test that negative default_sigma raises ValueError."""
        with pytest.raises(ValueError, match="default_sigma must be positive"):
            LinearBandit(default_sigma=-1.0)

    def test_invalid_obs_noise(self):
        """Test that negative obs_noise raises ValueError."""
        with pytest.raises(ValueError, match="obs_noise must be positive"):
            LinearBandit(obs_noise=0.0)


class TestLinearBanditUpdate:
    """Test LinearBandit update functionality."""

    def test_update_single_action_list(self):
        """Test update with a list of rewards for a single action."""
        bandit = LinearBandit()
        obs = {"arm_1": [1.0, 2.0, 3.0]}

        bandit.update(obs)

        assert "arm_1" in bandit.mu
        assert bandit.counts["arm_1"] == 3
        assert bandit.sum_rewards["arm_1"] == 6.0
        # Mean should be close to observed mean (2.0)
        assert 1.5 <= bandit.mu["arm_1"] <= 2.5

    def test_update_single_action_scalar(self):
        """Test update with a single scalar reward."""
        bandit = LinearBandit()
        obs = {"arm_1": 1.5}

        bandit.update(obs)

        assert "arm_1" in bandit.mu
        assert bandit.counts["arm_1"] == 1
        assert bandit.sum_rewards["arm_1"] == 1.5

    def test_update_multiple_actions(self):
        """Test update with multiple actions simultaneously."""
        bandit = LinearBandit()
        obs = {
            "arm_1": [1.0, 1.2, 1.1],
            "arm_2": [2.0, 2.1],
            "arm_3": 3.0
        }

        bandit.update(obs)

        assert len(bandit.mu) == 3
        assert bandit.counts["arm_1"] == 3
        assert bandit.counts["arm_2"] == 2
        assert bandit.counts["arm_3"] == 1

    def test_update_reduces_uncertainty(self):
        """Test that updating reduces posterior uncertainty."""
        bandit = LinearBandit(default_sigma=10.0)
        obs = {"arm_1": [1.0] * 100}  # Many observations

        initial_sigma = bandit.default_sigma
        bandit.update(obs)

        assert bandit.sigma["arm_1"] < initial_sigma

    def test_sequential_updates(self):
        """Test that sequential updates accumulate correctly."""
        bandit = LinearBandit()

        bandit.update({"arm_1": [1.0, 2.0]})
        assert bandit.counts["arm_1"] == 2

        bandit.update({"arm_1": [3.0]})
        assert bandit.counts["arm_1"] == 3
        assert bandit.sum_rewards["arm_1"] == 6.0

    def test_update_with_numpy_array(self):
        """Test update with numpy array of rewards."""
        bandit = LinearBandit()
        obs = {"arm_1": np.array([1.0, 2.0, 3.0])}

        bandit.update(obs)

        assert bandit.counts["arm_1"] == 3
        assert np.isclose(bandit.sum_rewards["arm_1"], 6.0)


class TestLinearBanditInputValidation:
    """Test input validation for LinearBandit."""

    def test_update_empty_obs(self):
        """Test that empty observations raise ValueError."""
        bandit = LinearBandit()
        with pytest.raises(ValueError, match="obs dictionary cannot be empty"):
            bandit.update({})

    def test_update_invalid_decay(self):
        """Test that invalid decay raises ValueError."""
        bandit = LinearBandit()
        obs = {"arm_1": [1.0]}

        with pytest.raises(ValueError, match="decay must be between 0.0 and 1.0"):
            bandit.update(obs, decay=-0.1)

        with pytest.raises(ValueError, match="decay must be between 0.0 and 1.0"):
            bandit.update(obs, decay=1.5)

    def test_update_invalid_reward_type(self):
        """Test that invalid reward types raise ValueError."""
        bandit = LinearBandit()
        obs = {"arm_1": "invalid"}

        with pytest.raises(ValueError, match="Rewards must be a number"):
            bandit.update(obs)

    def test_win_prop_invalid_draw(self):
        """Test that invalid draw parameter raises ValueError."""
        bandit = LinearBandit()
        bandit.update({"arm_1": [1.0]})

        with pytest.raises(ValueError, match="draw must be positive"):
            bandit.win_prop(draw=0)

        with pytest.raises(ValueError, match="draw must be positive"):
            bandit.win_prop(draw=-100)

    def test_win_prop_invalid_aggressive(self):
        """Test that invalid aggressive parameter raises ValueError."""
        bandit = LinearBandit()
        bandit.update({"arm_1": [1.0]})

        with pytest.raises(ValueError, match="aggressive must be positive"):
            bandit.win_prop(aggressive=0.0)


class TestLinearBanditWinProp:
    """Test win_prop (Thompson Sampling) functionality."""

    def test_win_prop_empty_actions(self):
        """Test win_prop with no actions returns empty dict."""
        bandit = LinearBandit()
        assert bandit.win_prop() == {}

    def test_win_prop_single_action(self):
        """Test win_prop with single action returns probability 1."""
        bandit = LinearBandit()
        bandit.update({"arm_1": [1.0]})

        probs = bandit.win_prop()
        assert probs == {"arm_1": 1.0}

    def test_win_prop_probabilities_sum_to_one(self):
        """Test that win probabilities sum to 1."""
        bandit = LinearBandit()
        bandit.update({
            "arm_1": [1.0, 1.1, 1.2],
            "arm_2": [2.0, 2.1, 2.2],
            "arm_3": [0.5, 0.6, 0.7]
        })

        probs = bandit.win_prop(draw=10000)
        assert np.isclose(sum(probs.values()), 1.0, atol=1e-10)

    def test_win_prop_better_action_wins(self):
        """Test that action with clearly better rewards has higher win prob."""
        bandit = LinearBandit(obs_noise=0.1)  # Low noise for clearer signal
        bandit.update({
            "arm_1": [1.0] * 100,  # Mean = 1.0
            "arm_2": [5.0] * 100,  # Mean = 5.0 (clearly better)
        })

        probs = bandit.win_prop(draw=50000)
        assert probs["arm_2"] > 0.95  # arm_2 should win almost always

    def test_win_prop_equal_actions(self):
        """Test that equal actions have similar win probabilities."""
        bandit = LinearBandit()
        bandit.update({
            "arm_1": [1.0] * 50,
            "arm_2": [1.0] * 50,
        })

        probs = bandit.win_prop(draw=50000)
        # Should be close to 0.5 each
        assert 0.4 < probs["arm_1"] < 0.6
        assert 0.4 < probs["arm_2"] < 0.6

    def test_win_prop_with_aggressive_parameter(self):
        """Test that aggressive parameter increases exploitation."""
        bandit = LinearBandit()
        bandit.update({
            "arm_1": [1.0] * 10,
            "arm_2": [1.5] * 10,
        })

        probs_normal = bandit.win_prop(draw=50000, aggressive=1.0)
        probs_aggressive = bandit.win_prop(draw=50000, aggressive=5.0)

        # With higher aggressive, the better arm should have even higher probability
        assert probs_aggressive["arm_2"] > probs_normal["arm_2"]

    def test_win_prop_subset_actions(self):
        """Test win_prop with a subset of actions."""
        bandit = LinearBandit()
        bandit.update({
            "arm_1": [1.0],
            "arm_2": [2.0],
            "arm_3": [3.0]
        })

        probs = bandit.win_prop(action_list=["arm_1", "arm_2"])
        assert "arm_3" not in probs
        assert "arm_1" in probs
        assert "arm_2" in probs

    def test_win_prop_unobserved_actions(self):
        """Test win_prop with unobserved actions included."""
        bandit = LinearBandit()
        bandit.update({"arm_1": [1.0]})

        probs = bandit.win_prop(action_list=["arm_1", "arm_2", "arm_3"])

        # All three should have positive probability
        assert all(p > 0 for p in probs.values())
        assert np.isclose(sum(probs.values()), 1.0)


class TestLinearBanditDecay:
    """Test decay functionality."""

    def test_decay_increases_uncertainty(self):
        """Test that decay increases posterior uncertainty."""
        bandit = LinearBandit()
        bandit.update({"arm_1": [1.0] * 100})

        sigma_before = bandit.sigma["arm_1"]
        bandit.update({"arm_1": [1.0]}, decay=0.5)
        sigma_after = bandit.sigma["arm_1"]

        assert sigma_after > sigma_before

    def test_decay_pulls_mean_toward_zero(self):
        """Test that decay pulls mean toward zero (prior mean)."""
        bandit = LinearBandit()
        bandit.update({"arm_1": [5.0] * 100})

        mu_before = bandit.mu["arm_1"]
        bandit.update({"arm_1": [5.0]}, decay=0.9)
        mu_after = bandit.mu["arm_1"]

        # Mean should be closer to 0 after decay
        assert abs(mu_after) < abs(mu_before)


class TestLinearBanditGetStatistics:
    """Test get_statistics functionality."""

    def test_get_statistics_empty(self):
        """Test get_statistics with no data."""
        bandit = LinearBandit()
        stats = bandit.get_statistics()
        assert stats == {}

    def test_get_statistics_structure(self):
        """Test that get_statistics returns correct structure."""
        bandit = LinearBandit()
        bandit.update({"arm_1": [1.0, 2.0, 3.0]})

        stats = bandit.get_statistics()

        assert "arm_1" in stats
        assert "mu" in stats["arm_1"]
        assert "sigma" in stats["arm_1"]
        assert "count" in stats["arm_1"]
        assert "mean_reward" in stats["arm_1"]

    def test_get_statistics_values(self):
        """Test that get_statistics returns correct values."""
        bandit = LinearBandit()
        bandit.update({"arm_1": [1.0, 2.0, 3.0]})

        stats = bandit.get_statistics()

        assert stats["arm_1"]["count"] == 3
        assert np.isclose(stats["arm_1"]["mean_reward"], 2.0)


class TestLinearBanditGetModels:
    """Test get_models functionality."""

    def test_get_models_empty(self):
        """Test get_models with no actions."""
        bandit = LinearBandit()
        assert bandit.get_models() == []

    def test_get_models_returns_action_list(self):
        """Test that get_models returns correct action list."""
        bandit = LinearBandit()
        bandit.update({
            "arm_1": [1.0],
            "arm_2": [2.0],
            "arm_3": [3.0]
        })

        models = bandit.get_models()
        assert set(models) == {"arm_1", "arm_2", "arm_3"}


class TestLinearBanditEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_update_with_zero_observations(self):
        """Test update with empty reward list."""
        bandit = LinearBandit()
        bandit.update({"arm_1": []})

        # Should not create the action
        assert "arm_1" not in bandit.mu

    def test_very_small_sigma(self):
        """Test that very small sigma (high confidence) works correctly."""
        bandit = LinearBandit()
        # Many identical observations should lead to small sigma
        bandit.update({"arm_1": [1.0] * 10000})

        assert bandit.sigma["arm_1"] < 0.1

    def test_negative_rewards(self):
        """Test that negative rewards are handled correctly."""
        bandit = LinearBandit()
        bandit.update({"arm_1": [-1.0, -2.0, -3.0]})

        assert bandit.mu["arm_1"] < 0
        assert bandit.sum_rewards["arm_1"] == -6.0

    def test_large_variance_in_rewards(self):
        """Test handling of high-variance rewards."""
        bandit = LinearBandit()
        bandit.update({"arm_1": [-100.0, 100.0, -100.0, 100.0]})

        # Mean should be close to 0
        assert -10 < bandit.mu["arm_1"] < 10
        assert bandit.counts["arm_1"] == 4
