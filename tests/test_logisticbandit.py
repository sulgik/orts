"""Unit tests for LogisticBandit class."""

import pytest
import numpy as np
from logisticbandit import LogisticBandit


class TestLogisticBanditInitialization:
    """Test LogisticBandit initialization."""

    def test_default_initialization(self):
        """Test initialization with default parameters."""
        bandit = LogisticBandit()
        assert len(bandit.get_models()) == 0
        assert bandit.mu.shape == (0,)
        assert bandit.sigma_inv.shape == (0, 0)

    def test_initialization_with_parameters(self):
        """Test initialization with custom parameters."""
        mu = np.array([0.1, 0.2, 0.0])
        sigma_inv = np.eye(3)
        action_list = ["arm_1", "arm_2", "arm_3"]

        bandit = LogisticBandit(mu=mu, sigma_inv=sigma_inv, action_list=action_list)

        assert bandit.get_models() == action_list
        assert np.allclose(bandit.mu, mu)
        assert np.allclose(bandit.sigma_inv, sigma_inv)


class TestLogisticBanditUpdate:
    """Test LogisticBandit update method."""

    def test_first_update(self):
        """Test first update with new observations."""
        bandit = LogisticBandit()
        obs = {"arm_1": [1000, 50], "arm_2": [1000, 45]}

        bandit.update(obs)

        assert "arm_1" in bandit.get_models()
        assert "arm_2" in bandit.get_models()
        # Check that parameters are initialized
        assert len(bandit.mu) >= 2  # At least 2 arms
        assert bandit.sigma_inv.shape[0] == len(bandit.mu)

    def test_update_with_new_arm(self):
        """Test adding a new arm to existing model."""
        bandit = LogisticBandit()

        # First update
        obs1 = {"arm_1": [1000, 50], "arm_2": [1000, 45]}
        bandit.update(obs1)

        # Second update with new arm
        obs2 = {"arm_1": [1000, 48], "arm_2": [1000, 47], "arm_3": [1000, 52]}
        bandit.update(obs2)

        assert len(bandit.get_models()) == 3
        assert "arm_3" in bandit.get_models()

    def test_update_odds_ratios_only(self):
        """Test update with ORTS (odds_ratios_only=True)."""
        bandit = LogisticBandit()
        obs = {"arm_1": [1000, 50], "arm_2": [1000, 45]}

        bandit.update(obs, odds_ratios_only=True)

        # Should complete without error
        assert len(bandit.get_models()) == 2

    def test_update_full_rank(self):
        """Test update with Full-TS (odds_ratios_only=False)."""
        bandit = LogisticBandit()
        obs = {"arm_1": [1000, 50], "arm_2": [1000, 45]}

        bandit.update(obs, odds_ratios_only=False)

        # Should complete without error
        assert len(bandit.get_models()) == 2

    def test_update_with_decay(self):
        """Test update with decay parameter."""
        bandit = LogisticBandit()
        obs = {"arm_1": [1000, 50], "arm_2": [1000, 45]}

        bandit.update(obs)
        mu_before = bandit.mu.copy()

        # Update with decay
        obs2 = {"arm_1": [1000, 60], "arm_2": [1000, 40]}
        bandit.update(obs2, decay=0.5)

        # Mu should have changed
        assert not np.allclose(bandit.mu, mu_before)


class TestLogisticBanditInputValidation:
    """Test input validation for LogisticBandit."""

    def test_update_empty_obs(self):
        """Test that empty observations raise ValueError."""
        bandit = LogisticBandit()

        with pytest.raises(ValueError, match="obs dictionary cannot be empty"):
            bandit.update({})

    def test_update_invalid_decay(self):
        """Test that invalid decay raises ValueError."""
        bandit = LogisticBandit()
        obs = {"arm_1": [1000, 50]}

        with pytest.raises(ValueError, match="decay must be between 0.0 and 1.0"):
            bandit.update(obs, decay=-0.1)

        with pytest.raises(ValueError, match="decay must be between 0.0 and 1.0"):
            bandit.update(obs, decay=1.5)

    def test_update_invalid_observation_format(self):
        """Test that invalid observation format raises ValueError."""
        bandit = LogisticBandit()

        # Wrong number of elements
        with pytest.raises(ValueError, match="Each observation must be"):
            bandit.update({"arm_1": [1000]})

        with pytest.raises(ValueError, match="Each observation must be"):
            bandit.update({"arm_1": [1000, 50, 30]})

    def test_update_negative_counts(self):
        """Test that negative counts raise ValueError."""
        bandit = LogisticBandit()

        with pytest.raises(ValueError, match="Total count must be non-negative"):
            bandit.update({"arm_1": [-1000, 50]})

        with pytest.raises(ValueError, match="Success count must be non-negative"):
            bandit.update({"arm_1": [1000, -50]})

    def test_update_success_exceeds_total(self):
        """Test that success > total raises ValueError."""
        bandit = LogisticBandit()

        with pytest.raises(ValueError, match="Success count .* cannot exceed total count"):
            bandit.update({"arm_1": [1000, 1500]})

    def test_win_prop_invalid_draw(self):
        """Test that invalid draw parameter raises ValueError."""
        bandit = LogisticBandit()
        obs = {"arm_1": [1000, 50], "arm_2": [1000, 45]}
        bandit.update(obs)

        with pytest.raises(ValueError, match="draw must be positive"):
            bandit.win_prop(draw=0)

        with pytest.raises(ValueError, match="draw must be positive"):
            bandit.win_prop(draw=-100)

    def test_win_prop_invalid_aggressive(self):
        """Test that invalid aggressive parameter raises ValueError."""
        bandit = LogisticBandit()
        obs = {"arm_1": [1000, 50], "arm_2": [1000, 45]}
        bandit.update(obs)

        with pytest.raises(ValueError, match="aggressive must be positive"):
            bandit.win_prop(aggressive=0)

        with pytest.raises(ValueError, match="aggressive must be positive"):
            bandit.win_prop(aggressive=-1.5)


class TestLogisticBanditWinProp:
    """Test LogisticBandit win_prop method."""

    def test_win_prop_empty(self):
        """Test win_prop with no arms."""
        bandit = LogisticBandit()
        result = bandit.win_prop()
        assert result == {}

    def test_win_prop_single_arm(self):
        """Test win_prop with single arm."""
        bandit = LogisticBandit()
        obs = {"arm_1": [1000, 50]}
        bandit.update(obs)

        result = bandit.win_prop()
        assert len(result) == 1
        assert result["arm_1"] == 1.0

    def test_win_prop_two_arms(self):
        """Test win_prop with two arms."""
        bandit = LogisticBandit()
        obs = {"arm_1": [1000, 50], "arm_2": [1000, 45]}
        bandit.update(obs)

        result = bandit.win_prop(draw=10000)

        # Check probabilities sum to 1
        assert abs(sum(result.values()) - 1.0) < 0.01

        # arm_1 should have higher probability (more successes)
        assert result["arm_1"] > result["arm_2"]

        # Both should have non-zero probability
        assert result["arm_1"] > 0
        assert result["arm_2"] > 0

    def test_win_prop_multiple_arms(self):
        """Test win_prop with multiple arms."""
        bandit = LogisticBandit()
        obs = {
            "arm_1": [1000, 50],
            "arm_2": [1000, 45],
            "arm_3": [1000, 55],
        }
        bandit.update(obs)

        result = bandit.win_prop(draw=10000)

        # Check probabilities sum to 1
        assert abs(sum(result.values()) - 1.0) < 0.01

        # arm_3 should have highest probability
        assert result["arm_3"] > result["arm_1"]
        assert result["arm_3"] > result["arm_2"]

    def test_win_prop_aggressive_parameter(self):
        """Test that aggressive parameter affects results."""
        bandit = LogisticBandit()
        obs = {"arm_1": [1000, 55], "arm_2": [1000, 45]}
        bandit.update(obs)

        result_normal = bandit.win_prop(aggressive=1.0, draw=10000)
        result_aggressive = bandit.win_prop(aggressive=3.0, draw=10000)

        # More aggressive should increase difference
        diff_normal = result_normal["arm_1"] - result_normal["arm_2"]
        diff_aggressive = result_aggressive["arm_1"] - result_aggressive["arm_2"]

        assert diff_aggressive > diff_normal

    def test_win_prop_subset_actions(self):
        """Test win_prop with subset of actions."""
        bandit = LogisticBandit()
        obs = {
            "arm_1": [1000, 50],
            "arm_2": [1000, 45],
            "arm_3": [1000, 40],
        }
        bandit.update(obs)

        result = bandit.win_prop(action_list=["arm_1", "arm_2"])

        assert len(result) == 2
        assert "arm_1" in result
        assert "arm_2" in result
        assert "arm_3" not in result


class TestLogisticBanditGetPar:
    """Test LogisticBandit get_par method."""

    def test_get_par_empty_list(self):
        """Test get_par with empty action list."""
        bandit = LogisticBandit()
        obs = {"arm_1": [1000, 50], "arm_2": [1000, 45]}
        bandit.update(obs)

        mu, sigma_inv = bandit.get_par([])
        assert mu is None
        assert sigma_inv is None

    def test_get_par_none(self):
        """Test get_par with None."""
        bandit = LogisticBandit()
        obs = {"arm_1": [1000, 50], "arm_2": [1000, 45]}
        bandit.update(obs)

        mu, sigma_inv = bandit.get_par(None)
        assert mu is None
        assert sigma_inv is None

    def test_get_par_valid_subset(self):
        """Test get_par with valid action subset."""
        bandit = LogisticBandit()
        obs = {"arm_1": [1000, 50], "arm_2": [1000, 45], "arm_3": [1000, 40]}
        bandit.update(obs)

        mu, sigma_inv = bandit.get_par(["arm_1", "arm_2"])

        assert mu is not None
        assert sigma_inv is not None
        assert len(mu) == 2
        assert sigma_inv.shape == (2, 2)


class TestLogisticBanditEdgeCases:
    """Test edge cases for LogisticBandit."""

    def test_zero_trials(self):
        """Test with zero trials (should be filtered out)."""
        bandit = LogisticBandit()
        obs = {"arm_1": [0, 0], "arm_2": [1000, 45]}
        bandit.update(obs)

        # arm_1 should be filtered out
        models = bandit.get_models()
        assert "arm_2" in models
        # arm_1 might not be in models or have zero contribution

    def test_perfect_success(self):
        """Test with 100% success rate."""
        bandit = LogisticBandit()
        obs = {"arm_1": [100, 100], "arm_2": [100, 50]}
        bandit.update(obs)

        result = bandit.win_prop(draw=10000)
        # arm_1 should have very high probability
        assert result["arm_1"] > 0.9

    def test_zero_success(self):
        """Test with 0% success rate."""
        bandit = LogisticBandit()
        obs = {"arm_1": [100, 0], "arm_2": [100, 50]}
        bandit.update(obs)

        result = bandit.win_prop(draw=10000)
        # arm_2 should have very high probability
        assert result["arm_2"] > 0.9

    def test_large_sample_sizes(self):
        """Test with large sample sizes."""
        bandit = LogisticBandit()
        obs = {"arm_1": [100000, 5000], "arm_2": [100000, 4900]}

        # Should handle large numbers without overflow
        bandit.update(obs)
        result = bandit.win_prop(draw=1000)

        assert abs(sum(result.values()) - 1.0) < 0.01


class TestLogisticBanditSequentialUpdates:
    """Test sequential updates to LogisticBandit."""

    def test_multiple_sequential_updates(self):
        """Test multiple sequential updates."""
        bandit = LogisticBandit()

        # First update
        obs1 = {"arm_1": [1000, 50], "arm_2": [1000, 45]}
        bandit.update(obs1)

        # Second update
        obs2 = {"arm_1": [1000, 55], "arm_2": [1000, 40]}
        bandit.update(obs2)

        # Third update
        obs3 = {"arm_1": [1000, 52], "arm_2": [1000, 48]}
        bandit.update(obs3)

        # Should still have 2 arms
        assert len(bandit.get_models()) == 2

        # Should still produce valid probabilities
        result = bandit.win_prop(draw=1000)
        assert abs(sum(result.values()) - 1.0) < 0.01

    def test_arm_removal_simulation(self):
        """Test behavior when an arm stops appearing."""
        bandit = LogisticBandit()

        # Start with 3 arms
        obs1 = {"arm_1": [1000, 50], "arm_2": [1000, 45], "arm_3": [1000, 40]}
        bandit.update(obs1)

        # Only update 2 arms (arm_3 stops being used)
        obs2 = {"arm_1": [1000, 52], "arm_2": [1000, 48]}
        bandit.update(obs2)

        # All 3 should still be tracked
        assert len(bandit.get_models()) == 3

        # Can still compute probabilities for all
        result = bandit.win_prop()
        assert len(result) == 3
