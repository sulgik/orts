"""Unit tests for utils module."""

import pytest
import numpy as np
from utils import logistic, is_pos_semidef, estimate


class TestLogistic:
    """Test logistic function."""

    def test_logistic_zero(self):
        """Test logistic at zero."""
        result = logistic(0.0)
        assert abs(result - 0.5) < 1e-10

    def test_logistic_positive(self):
        """Test logistic for positive values."""
        result = logistic(2.0)
        assert 0.5 < result < 1.0

        result = logistic(5.0)
        assert result > 0.99

    def test_logistic_negative(self):
        """Test logistic for negative values."""
        result = logistic(-2.0)
        assert 0.0 < result < 0.5

        result = logistic(-5.0)
        assert result < 0.01

    def test_logistic_symmetry(self):
        """Test that logistic(-x) = 1 - logistic(x)."""
        x = 2.5
        pos_result = logistic(x)
        neg_result = logistic(-x)
        assert abs(pos_result + neg_result - 1.0) < 1e-10

    def test_logistic_large_values(self):
        """Test logistic truncation for large values."""
        # Should truncate and not overflow
        result = logistic(100.0)
        assert 0.0 <= result <= 1.0
        assert result > 0.999

        result = logistic(-100.0)
        assert 0.0 <= result <= 1.0
        assert result < 0.001

    def test_logistic_custom_truncation(self):
        """Test logistic with custom truncation parameter."""
        result = logistic(5.0, trunc=3.0)
        # With trunc=3, values beyond ±3 should be clipped
        assert 0.0 <= result <= 1.0

    def test_logistic_monotonic(self):
        """Test that logistic is monotonically increasing."""
        x_values = np.linspace(-10, 10, 21)
        y_values = [logistic(x) for x in x_values]

        # Check monotonicity (with tolerance for floating point)
        for i in range(len(y_values) - 1):
            assert y_values[i] <= y_values[i + 1]  # Allow equal due to truncation


class TestIsPosSeimdef:
    """Test is_pos_semidef function."""

    def test_identity_matrix(self):
        """Test that identity matrix is positive semidefinite."""
        I = np.eye(3)
        assert is_pos_semidef(I) == True

    def test_diagonal_matrix(self):
        """Test diagonal matrices."""
        # Positive diagonal
        D = np.diag([1.0, 2.0, 3.0])
        assert is_pos_semidef(D) == True

        # Zero diagonal (positive semidefinite)
        D_zero = np.diag([0.0, 0.0, 0.0])
        assert is_pos_semidef(D_zero) == True

        # Negative diagonal
        D_neg = np.diag([1.0, -1.0, 3.0])
        assert is_pos_semidef(D_neg) == False

    def test_symmetric_positive_definite(self):
        """Test symmetric positive definite matrix."""
        A = np.array([
            [2.0, 1.0],
            [1.0, 2.0]
        ])
        assert is_pos_semidef(A) == True

    def test_non_positive_definite(self):
        """Test non-positive definite matrix."""
        A = np.array([
            [1.0, 2.0],
            [2.0, 1.0]
        ])
        # This matrix has eigenvalues [3, -1], so not positive semidefinite
        assert is_pos_semidef(A) == False

    def test_singular_matrix(self):
        """Test singular matrix (can be positive semidefinite)."""
        A = np.array([
            [1.0, 1.0],
            [1.0, 1.0]
        ])
        # This matrix has eigenvalues [2, 0], so it's positive semidefinite
        assert is_pos_semidef(A) == True

    def test_larger_matrix(self):
        """Test larger matrices."""
        # Create a clearly positive definite matrix
        A = np.array([
            [4.0, 1.0, 1.0],
            [1.0, 3.0, 0.5],
            [1.0, 0.5, 2.0]
        ])
        assert is_pos_semidef(A) == True


class TestEstimate:
    """Test estimate function through LogisticBandit integration."""

    def test_estimate_integration(self):
        """Test estimate function indirectly through LogisticBandit."""
        # estimate() is complex and depends on internal representation
        # Test it through the public API instead
        from logisticbandit import LogisticBandit

        bandit = LogisticBandit()
        obs = {"arm_1": [1000, 100], "arm_2": [1000, 50]}
        bandit.update(obs)

        # If estimate works, update should complete without error
        assert len(bandit.get_models()) == 2
        assert bandit.mu is not None
        assert bandit.sigma_inv is not None


class TestUtilsIntegration:
    """Integration tests for utils functions."""

    def test_logistic_inverse_relationship(self):
        """Test relationship between logistic and log odds."""
        p = 0.7
        log_odds = np.log(p / (1 - p))
        p_recovered = logistic(log_odds)

        assert abs(p - p_recovered) < 1e-10
