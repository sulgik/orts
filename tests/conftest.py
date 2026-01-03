"""Pytest configuration and fixtures for LogisticBandit tests."""

import pytest
import numpy as np
from logisticbandit import LogisticBandit
from ts import TSPar


@pytest.fixture
def simple_obs():
    """Fixture providing simple observations for testing."""
    return {"arm_1": [1000, 50], "arm_2": [1000, 45]}


@pytest.fixture
def three_arm_obs():
    """Fixture providing three-arm observations."""
    return {
        "arm_1": [1000, 50],
        "arm_2": [1000, 45],
        "arm_3": [1000, 55],
    }


@pytest.fixture
def unbalanced_obs():
    """Fixture providing unbalanced observations."""
    return {
        "arm_1": [1000, 80],
        "arm_2": [1000, 20],
    }


@pytest.fixture
def logistic_bandit():
    """Fixture providing a fresh LogisticBandit instance."""
    return LogisticBandit()


@pytest.fixture
def initialized_logistic_bandit(simple_obs):
    """Fixture providing an initialized LogisticBandit."""
    bandit = LogisticBandit()
    bandit.update(simple_obs)
    return bandit


@pytest.fixture
def ts_bandit():
    """Fixture providing a fresh TSPar instance."""
    return TSPar()


@pytest.fixture
def initialized_ts_bandit(simple_obs):
    """Fixture providing an initialized TSPar."""
    ts = TSPar()
    ts.update(simple_obs)
    return ts


@pytest.fixture
def random_seed():
    """Fixture to set random seed for reproducible tests."""
    np.random.seed(42)
    yield
    # Reset random state after test
    np.random.seed(None)


@pytest.fixture
def large_sample_obs():
    """Fixture providing large sample observations."""
    return {
        "arm_1": [100000, 5000],
        "arm_2": [100000, 4900],
    }


@pytest.fixture
def extreme_obs():
    """Fixture providing extreme success rates."""
    return {
        "arm_1": [100, 99],  # 99% success
        "arm_2": [100, 1],   # 1% success
    }


# Custom markers
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: mark test as slow to run"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
