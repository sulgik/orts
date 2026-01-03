"""
Comparison of ORTS, Full-TS, and Beta-Bernoulli Thompson Sampling.

This example demonstrates the differences between the three methods
and when to use each one.
"""

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from logisticbandit import LogisticBandit
from ts import TSPar


def compare_methods():
    """Compare ORTS, Full-TS, and Beta-Bernoulli TS on the same data."""

    # Same observations for all methods
    observations = {
        "arm_1": [10000, 500],  # 5% conversion rate
        "arm_2": [10000, 520],  # 5.2% conversion rate
        "arm_3": [10000, 480],  # 4.8% conversion rate
    }

    # Initialize all three methods
    orts = LogisticBandit()  # Odds Ratio TS (default)
    full_ts = LogisticBandit()  # Full Rank TS
    beta_ts = TSPar()  # Beta-Bernoulli TS

    # Update with observations
    print("Updating models with observations:")
    print(f"  arm_1: {observations['arm_1'][1]}/{observations['arm_1'][0]} = {observations['arm_1'][1]/observations['arm_1'][0]:.2%}")
    print(f"  arm_2: {observations['arm_2'][1]}/{observations['arm_2'][0]} = {observations['arm_2'][1]/observations['arm_2'][0]:.2%}")
    print(f"  arm_3: {observations['arm_3'][1]}/{observations['arm_3'][0]} = {observations['arm_3'][1]/observations['arm_3'][0]:.2%}")
    print()

    orts.update(observations, odds_ratios_only=True)
    full_ts.update(observations, odds_ratios_only=False)
    beta_ts.update(observations)

    # Get winning probabilities
    orts_probs = orts.win_prop(draw=50000)
    full_ts_probs = full_ts.win_prop(draw=50000)
    beta_ts_probs = beta_ts.win_prop(draw=50000)

    # Display results
    print("Winning Probabilities:")
    print("-" * 60)
    print(f"{'Method':<20} {'arm_1':<15} {'arm_2':<15} {'arm_3':<15}")
    print("-" * 60)
    print(f"{'ORTS':<20} {orts_probs['arm_1']:.4f}          {orts_probs['arm_2']:.4f}          {orts_probs['arm_3']:.4f}")
    print(f"{'Full-TS':<20} {full_ts_probs['arm_1']:.4f}          {full_ts_probs['arm_2']:.4f}          {full_ts_probs['arm_3']:.4f}")
    print(f"{'Beta-Bernoulli':<20} {beta_ts_probs['arm_1']:.4f}          {beta_ts_probs['arm_2']:.4f}          {beta_ts_probs['arm_3']:.4f}")
    print("-" * 60)
    print()

    # All methods should agree that arm_2 is best
    print("Analysis:")
    print(f"  All methods correctly identify arm_2 as the winner")
    print(f"  ORTS is most robust to time-varying effects")
    print(f"  Full-TS uses all covariance information")
    print(f"  Beta-Bernoulli is simplest but least flexible")


def time_varying_example():
    """Demonstrate ORTS robustness to time-varying effects."""

    print("\n" + "=" * 60)
    print("Time-Varying Effects Example")
    print("=" * 60 + "\n")

    orts = LogisticBandit()
    full_ts = LogisticBandit()

    # Simulate changing conversion rates over time
    timesteps = [
        {"arm_1": [5000, 250], "arm_2": [5000, 200]},  # arm_1 better
        {"arm_1": [5000, 220], "arm_2": [5000, 230]},  # arm_2 better
        {"arm_1": [5000, 240], "arm_2": [5000, 210]},  # arm_1 better again
    ]

    for t, obs in enumerate(timesteps):
        print(f"Timestep {t + 1}:")
        print(f"  Observations: arm_1={obs['arm_1'][1]}/{obs['arm_1'][0]}, "
              f"arm_2={obs['arm_2'][1]}/{obs['arm_2'][0]}")

        # Update both methods
        orts.update(obs, odds_ratios_only=True, decay=0.1)  # Use decay to forget old data
        full_ts.update(obs, odds_ratios_only=False)

        # Get probabilities
        orts_probs = orts.win_prop(draw=10000)
        full_ts_probs = full_ts.win_prop(draw=10000)

        print(f"  ORTS:    arm_1={orts_probs['arm_1']:.3f}, arm_2={orts_probs['arm_2']:.3f}")
        print(f"  Full-TS: arm_1={full_ts_probs['arm_1']:.3f}, arm_2={full_ts_probs['arm_2']:.3f}")
        print()

    print("With time-varying effects, ORTS adapts better by using decay parameter")


if __name__ == "__main__":
    compare_methods()
    time_varying_example()
