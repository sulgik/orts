"""
Linear (Gaussian) Bandit Example.

Demonstrates the use of LinearBandit for Thompson Sampling with continuous
rewards. This is appropriate for scenarios where rewards are continuous
values rather than binary outcomes.

Use cases:
- Ad click revenue optimization
- Server response time minimization
- Product rating optimization
- Price optimization with continuous revenue
"""

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from linearbandit import LinearBandit


def basic_usage():
    """Basic usage example of LinearBandit."""
    print("=" * 70)
    print("Basic LinearBandit Usage")
    print("=" * 70)
    print()

    # Initialize the bandit
    bandit = LinearBandit(obs_noise=1.0, default_sigma=2.0)

    # Simulate three advertising variants with different average revenues
    print("Simulating three ad variants with different average revenues:")
    print("  Variant A: Average revenue = $2.50")
    print("  Variant B: Average revenue = $3.20")
    print("  Variant C: Average revenue = $2.80")
    print()

    # Initial observations
    obs = {
        "variant_a": [2.3, 2.7, 2.4, 2.6, 2.5],  # Mean ≈ 2.5
        "variant_b": [3.1, 3.3, 3.0, 3.4, 3.2],  # Mean ≈ 3.2
        "variant_c": [2.7, 2.9, 2.8, 2.7, 2.9],  # Mean ≈ 2.8
    }

    bandit.update(obs)

    # Show statistics
    print("Statistics after initial observations:")
    print("-" * 70)
    stats = bandit.get_statistics()
    for action, stat in stats.items():
        print(f"{action:12s}: μ={stat['mu']:.3f}, σ={stat['sigma']:.3f}, "
              f"count={stat['count']}, mean={stat['mean_reward']:.3f}")
    print()

    # Calculate winning probabilities
    probs = bandit.win_prop(draw=100000)
    print("Winning Probabilities (Thompson Sampling):")
    print("-" * 70)
    for action, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
        print(f"{action:12s}: {prob:.4f} ({prob*100:.2f}%)")
    print()


def sequential_learning():
    """Demonstrate how the bandit learns over time."""
    print("=" * 70)
    print("Sequential Learning Example")
    print("=" * 70)
    print()

    bandit = LinearBandit(obs_noise=0.5)

    # True means (unknown to the bandit)
    true_means = {
        "option_1": 1.0,
        "option_2": 1.5,
        "option_3": 0.8,
    }

    print("True means:", true_means)
    print("Observing 10 rounds of data...\n")

    for round_num in range(1, 11):
        # Simulate observations with Gaussian noise
        obs = {}
        for action, true_mean in true_means.items():
            # Generate 5 observations per round
            obs[action] = list(np.random.normal(true_mean, 0.5, 5))

        bandit.update(obs)

        if round_num in [1, 3, 5, 10]:
            print(f"After round {round_num}:")
            probs = bandit.win_prop(draw=50000)
            for action in ["option_1", "option_2", "option_3"]:
                stats = bandit.get_statistics()[action]
                print(f"  {action}: μ={stats['mu']:.3f}±{stats['sigma']:.3f}, "
                      f"P(win)={probs[action]:.3f}")
            print()


def exploration_exploitation():
    """Demonstrate exploration vs exploitation with aggressive parameter."""
    print("=" * 70)
    print("Exploration vs Exploitation")
    print("=" * 70)
    print()

    bandit = LinearBandit(obs_noise=1.0)

    # Update with observations
    obs = {
        "action_a": [5.0] * 20,  # Mean = 5.0, low uncertainty
        "action_b": [5.2, 5.3],  # Mean ≈ 5.25, high uncertainty
    }
    bandit.update(obs)

    print("Scenario: action_a has more data (n=20), action_b has less (n=2)")
    print("          action_b appears slightly better but has high uncertainty")
    print()

    # Compare different aggressive values
    for aggressive in [0.5, 1.0, 2.0, 5.0]:
        probs = bandit.win_prop(draw=50000, aggressive=aggressive)
        print(f"aggressive={aggressive:.1f}: "
              f"action_a={probs['action_a']:.3f}, "
              f"action_b={probs['action_b']:.3f}")

    print()
    print("Higher 'aggressive' values increase exploitation of the better action.")
    print()


def dynamic_environment_with_decay():
    """Demonstrate decay for handling non-stationary environments."""
    print("=" * 70)
    print("Dynamic Environment with Decay")
    print("=" * 70)
    print()

    bandit = LinearBandit(obs_noise=0.5)

    # Phase 1: action_a is better
    print("Phase 1: action_a has mean=2.0, action_b has mean=1.0")
    for _ in range(5):
        obs = {
            "action_a": list(np.random.normal(2.0, 0.5, 10)),
            "action_b": list(np.random.normal(1.0, 0.5, 10)),
        }
        bandit.update(obs, decay=0.0)

    probs = bandit.win_prop(draw=50000)
    print(f"  P(action_a wins) = {probs['action_a']:.3f}")
    print()

    # Phase 2: Means switch! action_b is now better
    print("Phase 2: Means switch! action_a now=1.0, action_b now=2.5")
    print()

    # Without decay
    bandit_no_decay = LinearBandit(obs_noise=0.5)
    bandit_no_decay.mu = bandit.mu.copy()
    bandit_no_decay.sigma = bandit.sigma.copy()
    bandit_no_decay.counts = bandit.counts.copy()
    bandit_no_decay.sum_rewards = bandit.sum_rewards.copy()

    # With decay
    bandit_with_decay = LinearBandit(obs_noise=0.5)
    bandit_with_decay.mu = bandit.mu.copy()
    bandit_with_decay.sigma = bandit.sigma.copy()
    bandit_with_decay.counts = bandit.counts.copy()
    bandit_with_decay.sum_rewards = bandit.sum_rewards.copy()

    for round_num in range(1, 6):
        obs = {
            "action_a": list(np.random.normal(1.0, 0.5, 10)),
            "action_b": list(np.random.normal(2.5, 0.5, 10)),
        }

        bandit_no_decay.update(obs, decay=0.0)
        bandit_with_decay.update(obs, decay=0.3)

    probs_no_decay = bandit_no_decay.win_prop(draw=50000)
    probs_with_decay = bandit_with_decay.win_prop(draw=50000)

    print("After 5 rounds of switched data:")
    print(f"  Without decay: P(action_b wins) = {probs_no_decay['action_b']:.3f}")
    print(f"  With decay=0.3: P(action_b wins) = {probs_with_decay['action_b']:.3f}")
    print()
    print("Decay helps adapt faster to changing environments!")
    print()


def comparison_with_large_difference():
    """Example where one action is clearly superior."""
    print("=" * 70)
    print("Clear Winner Scenario")
    print("=" * 70)
    print()

    bandit = LinearBandit(obs_noise=0.3)  # Low noise for clearer signal

    # One action is clearly better
    obs = {
        "standard": [1.0] * 100,   # Mean = 1.0
        "premium": [5.0] * 100,    # Mean = 5.0 (much better!)
        "budget": [0.5] * 100,     # Mean = 0.5
    }

    bandit.update(obs)

    print("Three options with very different performance:")
    stats = bandit.get_statistics()
    for action, stat in stats.items():
        print(f"  {action:10s}: mean = {stat['mean_reward']:.2f}, "
              f"μ = {stat['mu']:.3f} ± {stat['sigma']:.4f}")
    print()

    probs = bandit.win_prop(draw=100000)
    print("Winning probabilities:")
    for action, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
        print(f"  {action:10s}: {prob:.6f} ({prob*100:.4f}%)")
    print()
    print("With enough data and low noise, Thompson Sampling correctly")
    print("identifies the best action with very high confidence.")
    print()


if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility

    basic_usage()
    sequential_learning()
    exploration_exploitation()
    dynamic_environment_with_decay()
    comparison_with_large_difference()

    print("=" * 70)
    print("All examples completed!")
    print("=" * 70)
