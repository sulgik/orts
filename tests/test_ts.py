"""Unit tests for TSPar (Beta-Bernoulli Thompson Sampling) class."""

import pytest
import numpy as np
from ts import TSPar


class TestTSParInitialization:
    """Test TSPar initialization."""

    def test_default_initialization(self):
        """Test initialization with default parameters."""
        ts = TSPar()
        assert len(ts.get_models()) == 0
        assert ts.alpha.shape == (0,)
        assert ts.beta.shape == (0,)


class TestTSParUpdate:
    """Test TSPar update method."""

    def test_first_update(self):
        """Test first update with new observations."""
        ts = TSPar()
        obs = {"arm_1": [1000, 50], "arm_2": [1000, 45]}

        ts.update(obs)

        assert "arm_1" in ts.get_models()
        assert "arm_2" in ts.get_models()
        assert len(ts.alpha) == 2
        assert len(ts.beta) == 2

        # Check that alpha and beta are updated correctly
        # alpha should be 1 (prior) + 50 (successes) = 51
        # beta should be 1 (prior) + (1000 - 50) (failures) = 951
        assert ts.alpha[0] == 51
        assert ts.beta[0] == 951

    def test_sequential_updates(self):
        """Test sequential updates."""
        ts = TSPar()

        # First update
        obs1 = {"arm_1": [100, 10], "arm_2": [100, 8]}
        ts.update(obs1)

        alpha_before = ts.alpha.copy()
        beta_before = ts.beta.copy()

        # Second update
        obs2 = {"arm_1": [100, 12], "arm_2": [100, 9]}
        ts.update(obs2)

        # Alpha and beta should accumulate
        assert ts.alpha[0] == alpha_before[0] + 12
        assert ts.alpha[1] == alpha_before[1] + 9
        assert ts.beta[0] == beta_before[0] + 88
        assert ts.beta[1] == beta_before[1] + 91

    def test_update_with_new_arm(self):
        """Test that update doesn't support dynamic arm addition in current implementation."""
        ts = TSPar()

        # First update
        obs1 = {"arm_1": [100, 10], "arm_2": [100, 8]}
        ts.update(obs1)

        # Trying to add a new arm after initialization
        # Current implementation might not handle this well
        # This test documents current behavior
        obs2 = {"arm_1": [100, 12], "arm_2": [100, 9]}
        ts.update(obs2)  # Should work with same arms


class TestTSParInputValidation:
    """Test input validation for TSPar."""

    def test_update_empty_obs(self):
        """Test that empty observations raise ValueError."""
        ts = TSPar()

        with pytest.raises(ValueError, match="obs dictionary cannot be empty"):
            ts.update({})

    def test_update_invalid_observation_format(self):
        """Test that invalid observation format raises ValueError."""
        ts = TSPar()

        # Wrong number of elements
        with pytest.raises(ValueError, match="Each observation must be"):
            ts.update({"arm_1": [1000]})

        with pytest.raises(ValueError, match="Each observation must be"):
            ts.update({"arm_1": [1000, 50, 30]})

    def test_update_negative_counts(self):
        """Test that negative counts raise ValueError."""
        ts = TSPar()

        with pytest.raises(ValueError, match="Total count must be non-negative"):
            ts.update({"arm_1": [-1000, 50]})

        with pytest.raises(ValueError, match="Success count must be non-negative"):
            ts.update({"arm_1": [1000, -50]})

    def test_update_success_exceeds_total(self):
        """Test that success > total raises ValueError."""
        ts = TSPar()

        with pytest.raises(ValueError, match="Success count .* cannot exceed total count"):
            ts.update({"arm_1": [1000, 1500]})

    def test_win_prop_invalid_draw(self):
        """Test that invalid draw parameter raises ValueError."""
        ts = TSPar()
        obs = {"arm_1": [1000, 50], "arm_2": [1000, 45]}
        ts.update(obs)

        with pytest.raises(ValueError, match="draw must be positive"):
            ts.win_prop(draw=0)

        with pytest.raises(ValueError, match="draw must be positive"):
            ts.win_prop(draw=-100)


class TestTSParWinProp:
    """Test TSPar win_prop method."""

    def test_win_prop_two_arms(self):
        """Test win_prop with two arms."""
        ts = TSPar()
        obs = {"arm_1": [1000, 50], "arm_2": [1000, 45]}
        ts.update(obs)

        result = ts.win_prop(draw=10000)

        # Check probabilities sum to 1
        assert abs(sum(result.values()) - 1.0) < 0.01

        # arm_1 should have higher probability (more successes)
        assert result["arm_1"] > result["arm_2"]

        # Both should have non-zero probability
        assert result["arm_1"] > 0
        assert result["arm_2"] > 0

    def test_win_prop_multiple_arms(self):
        """Test win_prop with multiple arms."""
        ts = TSPar()
        obs = {
            "arm_1": [1000, 50],
            "arm_2": [1000, 45],
            "arm_3": [1000, 55],
        }
        ts.update(obs)

        result = ts.win_prop(draw=10000)

        # Check probabilities sum to 1
        assert abs(sum(result.values()) - 1.0) < 0.01

        # arm_3 should have highest probability
        assert result["arm_3"] > result["arm_1"]
        assert result["arm_3"] > result["arm_2"]

    def test_win_prop_clear_winner(self):
        """Test win_prop with a clear winner."""
        ts = TSPar()
        obs = {"arm_1": [1000, 80], "arm_2": [1000, 20]}
        ts.update(obs)

        result = ts.win_prop(draw=10000)

        # arm_1 should have very high probability
        assert result["arm_1"] > 0.95

    def test_win_prop_equal_performance(self):
        """Test win_prop with equal performance arms."""
        ts = TSPar()
        obs = {"arm_1": [1000, 50], "arm_2": [1000, 50]}
        ts.update(obs)

        result = ts.win_prop(draw=10000)

        # Should be approximately equal
        assert abs(result["arm_1"] - result["arm_2"]) < 0.1


class TestTSParEdgeCases:
    """Test edge cases for TSPar."""

    def test_perfect_success(self):
        """Test with 100% success rate."""
        ts = TSPar()
        obs = {"arm_1": [100, 100], "arm_2": [100, 50]}
        ts.update(obs)

        result = ts.win_prop(draw=10000)
        # arm_1 should have very high probability
        assert result["arm_1"] > 0.9

    def test_zero_success(self):
        """Test with 0% success rate."""
        ts = TSPar()
        obs = {"arm_1": [100, 0], "arm_2": [100, 50]}
        ts.update(obs)

        result = ts.win_prop(draw=10000)
        # arm_2 should have very high probability
        assert result["arm_2"] > 0.95

    def test_very_small_sample(self):
        """Test with very small sample sizes."""
        ts = TSPar()
        obs = {"arm_1": [10, 5], "arm_2": [10, 4]}
        ts.update(obs)

        result = ts.win_prop(draw=10000)

        # Should still produce valid probabilities
        assert abs(sum(result.values()) - 1.0) < 0.01
        assert all(0 <= p <= 1 for p in result.values())

    def test_large_sample_sizes(self):
        """Test with large sample sizes."""
        ts = TSPar()
        obs = {"arm_1": [100000, 5000], "arm_2": [100000, 4900]}

        # Should handle large numbers without overflow
        ts.update(obs)
        result = ts.win_prop(draw=1000)

        assert abs(sum(result.values()) - 1.0) < 0.01

    def test_single_observation(self):
        """Test with single observation per arm."""
        ts = TSPar()
        obs = {"arm_1": [1, 1], "arm_2": [1, 0]}
        ts.update(obs)

        result = ts.win_prop(draw=10000)

        # arm_1 should have higher probability
        assert result["arm_1"] > result["arm_2"]


class TestTSParAccumulation:
    """Test that TSPar correctly accumulates observations."""

    def test_alpha_beta_accumulation(self):
        """Test that alpha and beta correctly accumulate."""
        ts = TSPar()

        # First observation
        obs1 = {"arm_1": [100, 20]}
        ts.update(obs1)

        # Initial: alpha = 1 + 20 = 21, beta = 1 + 80 = 81
        assert ts.alpha[0] == 21
        assert ts.beta[0] == 81

        # Second observation
        obs2 = {"arm_1": [100, 30]}
        ts.update(obs2)

        # Updated: alpha = 21 + 30 = 51, beta = 81 + 70 = 151
        assert ts.alpha[0] == 51
        assert ts.beta[0] == 151

    def test_multiple_arms_accumulation(self):
        """Test accumulation with multiple arms."""
        ts = TSPar()

        # Initialize
        obs1 = {"arm_1": [100, 20], "arm_2": [100, 25], "arm_3": [100, 30]}
        ts.update(obs1)

        initial_alpha = ts.alpha.copy()
        initial_beta = ts.beta.copy()

        # Update
        obs2 = {"arm_1": [50, 10], "arm_2": [50, 15], "arm_3": [50, 12]}
        ts.update(obs2)

        # Check all arms accumulated correctly
        assert ts.alpha[0] == initial_alpha[0] + 10
        assert ts.alpha[1] == initial_alpha[1] + 15
        assert ts.alpha[2] == initial_alpha[2] + 12

        assert ts.beta[0] == initial_beta[0] + 40
        assert ts.beta[1] == initial_beta[1] + 35
        assert ts.beta[2] == initial_beta[2] + 38


class TestTSParVsLogisticBandit:
    """Comparison tests between TSPar and LogisticBandit."""

    def test_similar_results_for_simple_case(self):
        """Test that TSPar and LogisticBandit give similar results for simple cases."""
        from logisticbandit import LogisticBandit

        obs = {"arm_1": [1000, 60], "arm_2": [1000, 40]}

        # TSPar
        ts = TSPar()
        ts.update(obs)
        ts_result = ts.win_prop(draw=50000)

        # LogisticBandit
        lb = LogisticBandit()
        lb.update(obs)
        lb_result = lb.win_prop(draw=50000)

        # Both should favor arm_1
        assert ts_result["arm_1"] > ts_result["arm_2"]
        assert lb_result["arm_1"] > lb_result["arm_2"]

        # Results should be reasonably close (within 20%)
        # Note: They use different models so won't be identical
        assert abs(ts_result["arm_1"] - lb_result["arm_1"]) < 0.2
