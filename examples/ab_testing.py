"""
A/B Testing with Thompson Sampling.

Realistic A/B testing scenario using ORTS for dynamic traffic allocation.
"""

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from logisticbandit import LogisticBandit


def simulate_ab_test(true_rates, n_users=10000, n_rounds=10):
    """
    Simulate an A/B test with Thompson Sampling for traffic allocation.

    Parameters
    ----------
    true_rates : dict
        True conversion rates for each variant
    n_users : int
        Total users per round
    n_rounds : int
        Number of rounds to run

    Returns
    -------
    dict
        Summary statistics
    """
    bandit = LogisticBandit()

    total_conversions = {arm: 0 for arm in true_rates.keys()}
    total_views = {arm: 0 for arm in true_rates.keys()}

    print(f"Starting A/B Test with {len(true_rates)} variants")
    print(f"True conversion rates: {true_rates}")
    print(f"Total users per round: {n_users:,}")
    print(f"Number of rounds: {n_rounds}")
    print("=" * 70)
    print()

    for round_num in range(1, n_rounds + 1):
        # Get allocation probabilities from Thompson Sampling
        if round_num == 1:
            # First round: equal allocation
            allocation = {arm: 1.0 / len(true_rates) for arm in true_rates.keys()}
        else:
            allocation = bandit.win_prop(draw=10000)

        # Allocate users based on Thompson Sampling probabilities
        observations = {}
        for arm, prob in allocation.items():
            n_allocated = int(n_users * prob)
            true_rate = true_rates[arm]

            # Simulate conversions (binomial sampling)
            n_conversions = np.random.binomial(n_allocated, true_rate)

            observations[arm] = [n_allocated, n_conversions]
            total_views[arm] += n_allocated
            total_conversions[arm] += n_conversions

        # Update Thompson Sampling model
        bandit.update(observations)

        # Print round summary
        print(f"Round {round_num}:")
        for arm in true_rates.keys():
            n_alloc, n_conv = observations[arm]
            conv_rate = n_conv / n_alloc if n_alloc > 0 else 0
            print(f"  {arm}: {n_alloc:5d} users ({allocation[arm]:5.1%}) → "
                  f"{n_conv:4d} conversions ({conv_rate:5.2%})")
        print()

    # Final summary
    print("=" * 70)
    print("FINAL RESULTS:")
    print("-" * 70)
    print(f"{'Variant':<15} {'Views':<12} {'Conversions':<15} {'Obs. Rate':<12} {'True Rate'}")
    print("-" * 70)

    for arm in true_rates.keys():
        obs_rate = total_conversions[arm] / total_views[arm] if total_views[arm] > 0 else 0
        print(f"{arm:<15} {total_views[arm]:<12,} {total_conversions[arm]:<15,} "
              f"{obs_rate:<12.2%} {true_rates[arm]:.2%}")

    # Calculate regret
    best_rate = max(true_rates.values())
    total_users = sum(total_views.values())
    optimal_conversions = total_users * best_rate
    actual_conversions = sum(total_conversions.values())
    regret = optimal_conversions - actual_conversions

    print("-" * 70)
    print(f"Total Users: {total_users:,}")
    print(f"Optimal Conversions: {optimal_conversions:,.0f}")
    print(f"Actual Conversions: {actual_conversions:,}")
    print(f"Regret: {regret:,.0f} ({regret/total_users:.2%})")
    print("=" * 70)

    return {
        "total_views": total_views,
        "total_conversions": total_conversions,
        "regret": regret,
        "regret_percent": regret / total_users
    }


def compare_to_equal_allocation():
    """Compare Thompson Sampling to equal allocation."""

    true_rates = {
        "control": 0.05,     # 5% conversion
        "variant_a": 0.052,  # 5.2% conversion (winner)
        "variant_b": 0.048,  # 4.8% conversion
    }

    print("\n" + "=" * 70)
    print("THOMPSON SAMPLING vs EQUAL ALLOCATION")
    print("=" * 70 + "\n")

    # Thompson Sampling
    print("METHOD 1: Thompson Sampling (Adaptive Allocation)")
    print("-" * 70)
    np.random.seed(42)
    ts_result = simulate_ab_test(true_rates, n_users=10000, n_rounds=10)

    # Equal Allocation (for comparison)
    print("\nMETHOD 2: Equal Allocation (Traditional A/B Test)")
    print("-" * 70)

    total_users = 100000
    users_per_variant = total_users // len(true_rates)

    equal_conversions = {}
    for arm, rate in true_rates.items():
        conversions = np.random.binomial(users_per_variant, rate)
        equal_conversions[arm] = conversions
        print(f"{arm}: {users_per_variant:,} users → {conversions:,} conversions "
              f"({conversions/users_per_variant:.2%})")

    best_rate = max(true_rates.values())
    equal_regret = (total_users * best_rate) - sum(equal_conversions.values())

    print("-" * 70)
    print(f"Equal Allocation Regret: {equal_regret:,.0f} ({equal_regret/total_users:.2%})")
    print("=" * 70)

    # Comparison
    print("\nCOMPARISON:")
    print(f"  Thompson Sampling Regret: {ts_result['regret']:,.0f} "
          f"({ts_result['regret_percent']:.2%})")
    print(f"  Equal Allocation Regret:  {equal_regret:,.0f} "
          f"({equal_regret/total_users:.2%})")
    print(f"  Improvement: {((equal_regret - ts_result['regret']) / equal_regret):.1%} less regret")
    print()
    print("Thompson Sampling allocates more traffic to better-performing variants,")
    print("reducing regret while still exploring all options.")


if __name__ == "__main__":
    # Example 1: Simple A/B test
    true_rates = {
        "control": 0.05,
        "variant_a": 0.06,
        "variant_b": 0.045,
    }

    simulate_ab_test(true_rates, n_users=5000, n_rounds=10)

    # Example 2: Compare to equal allocation
    compare_to_equal_allocation()
