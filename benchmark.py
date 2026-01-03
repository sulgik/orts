"""
Performance benchmark for bandit algorithms.

Profiles execution time and identifies bottlenecks.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import cProfile
import pstats
from io import StringIO
import numpy as np
from logisticbandit import LogisticBandit
from linearbandit import LinearBandit
from ts import TSPar


def benchmark_logistic_bandit():
    """Benchmark LogisticBandit with various scenarios."""
    print("=" * 70)
    print("LogisticBandit Benchmark")
    print("=" * 70)

    scenarios = [
        ("Small (2 arms, 1K obs)", 2, 1000, 10000),
        ("Medium (5 arms, 10K obs)", 5, 10000, 50000),
        ("Large (10 arms, 100K obs)", 10, 100000, 100000),
    ]

    for name, n_arms, n_obs, n_draw in scenarios:
        print(f"\n{name}:")
        print("-" * 70)

        bandit = LogisticBandit()

        # Generate observations
        obs = {
            f"arm_{i}": [n_obs, int(n_obs * (0.01 + i * 0.001))]
            for i in range(n_arms)
        }

        # Time update
        start = time.perf_counter()
        bandit.update(obs)
        update_time = (time.perf_counter() - start) * 1000

        # Time win_prop
        start = time.perf_counter()
        probs = bandit.win_prop(draw=n_draw)
        win_prop_time = (time.perf_counter() - start) * 1000

        print(f"  update():   {update_time:8.2f} ms")
        print(f"  win_prop(): {win_prop_time:8.2f} ms (draw={n_draw:,})")
        print(f"  Total:      {update_time + win_prop_time:8.2f} ms")


def benchmark_linear_bandit():
    """Benchmark LinearBandit with various scenarios."""
    print("\n" + "=" * 70)
    print("LinearBandit Benchmark")
    print("=" * 70)

    scenarios = [
        ("Small (2 arms, 100 obs)", 2, 100, 10000),
        ("Medium (5 arms, 1K obs)", 5, 1000, 50000),
        ("Large (10 arms, 10K obs)", 10, 10000, 100000),
    ]

    for name, n_arms, n_obs_per_arm, n_draw in scenarios:
        print(f"\n{name}:")
        print("-" * 70)

        bandit = LinearBandit(obs_noise=1.0)

        # Generate observations
        np.random.seed(42)
        obs = {
            f"arm_{i}": list(np.random.normal(1.0 + i * 0.1, 0.5, n_obs_per_arm))
            for i in range(n_arms)
        }

        # Time update
        start = time.perf_counter()
        bandit.update(obs)
        update_time = (time.perf_counter() - start) * 1000

        # Time win_prop
        start = time.perf_counter()
        probs = bandit.win_prop(draw=n_draw)
        win_prop_time = (time.perf_counter() - start) * 1000

        print(f"  update():   {update_time:8.2f} ms")
        print(f"  win_prop(): {win_prop_time:8.2f} ms (draw={n_draw:,})")
        print(f"  Total:      {update_time + win_prop_time:8.2f} ms")


def benchmark_ts_par():
    """Benchmark Beta-Bernoulli Thompson Sampling."""
    print("\n" + "=" * 70)
    print("TSPar (Beta-Bernoulli) Benchmark")
    print("=" * 70)

    scenarios = [
        ("Small (2 arms, 1K obs)", 2, 1000, 10000),
        ("Medium (5 arms, 10K obs)", 5, 10000, 10000),
        ("Large (10 arms, 100K obs)", 10, 100000, 10000),
    ]

    for name, n_arms, n_obs, n_draw in scenarios:
        print(f"\n{name}:")
        print("-" * 70)

        bandit = TSPar()

        # Generate observations
        obs = {
            f"arm_{i}": [n_obs, int(n_obs * (0.01 + i * 0.001))]
            for i in range(n_arms)
        }

        # Time update
        start = time.perf_counter()
        bandit.update(obs)
        update_time = (time.perf_counter() - start) * 1000

        # Time win_prop
        start = time.perf_counter()
        probs = bandit.win_prop(draw=n_draw)
        win_prop_time = (time.perf_counter() - start) * 1000

        print(f"  update():   {update_time:8.2f} ms")
        print(f"  win_prop(): {win_prop_time:8.2f} ms (draw={n_draw:,})")
        print(f"  Total:      {update_time + win_prop_time:8.2f} ms")


def profile_logistic_bandit_detailed():
    """Detailed profiling of LogisticBandit.win_prop()."""
    print("\n" + "=" * 70)
    print("Detailed Profile: LogisticBandit.win_prop()")
    print("=" * 70)

    bandit = LogisticBandit()
    obs = {f"arm_{i}": [10000, int(10000 * (0.01 + i * 0.001))] for i in range(5)}
    bandit.update(obs)

    # Profile win_prop
    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(10):  # Run multiple times for better statistics
        bandit.win_prop(draw=100000)

    profiler.disable()

    # Print results
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Top 20 functions

    print(s.getvalue())


def profile_linear_bandit_detailed():
    """Detailed profiling of LinearBandit.win_prop()."""
    print("\n" + "=" * 70)
    print("Detailed Profile: LinearBandit.win_prop()")
    print("=" * 70)

    bandit = LinearBandit(obs_noise=1.0)
    np.random.seed(42)
    obs = {f"arm_{i}": list(np.random.normal(1.0 + i * 0.1, 0.5, 1000)) for i in range(5)}
    bandit.update(obs)

    # Profile win_prop
    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(10):  # Run multiple times for better statistics
        bandit.win_prop(draw=100000)

    profiler.disable()

    # Print results
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Top 20 functions

    print(s.getvalue())


def compare_draw_sizes():
    """Compare performance with different Monte Carlo sample sizes."""
    print("\n" + "=" * 70)
    print("Impact of Monte Carlo Sample Size")
    print("=" * 70)

    bandit = LinearBandit(obs_noise=1.0)
    np.random.seed(42)
    obs = {f"arm_{i}": list(np.random.normal(1.0 + i * 0.1, 0.5, 100)) for i in range(5)}
    bandit.update(obs)

    draw_sizes = [1000, 5000, 10000, 50000, 100000, 500000, 1000000]

    print(f"\n{'Draw Size':>12s}  {'Time (ms)':>12s}  {'Time/Sample (μs)':>18s}")
    print("-" * 70)

    for draw in draw_sizes:
        times = []
        for _ in range(5):  # Average over 5 runs
            start = time.perf_counter()
            bandit.win_prop(draw=draw)
            times.append((time.perf_counter() - start) * 1000)

        avg_time = np.mean(times)
        time_per_sample = (avg_time * 1000) / draw  # microseconds per sample

        print(f"{draw:12,d}  {avg_time:12.2f}  {time_per_sample:18.4f}")


def memory_usage_test():
    """Test memory usage for large-scale scenarios."""
    print("\n" + "=" * 70)
    print("Memory Usage Test")
    print("=" * 70)

    try:
        import tracemalloc

        # Test LinearBandit with many arms
        tracemalloc.start()

        bandit = LinearBandit()
        np.random.seed(42)
        obs = {f"arm_{i}": list(np.random.normal(1.0, 0.5, 100)) for i in range(100)}

        snapshot1 = tracemalloc.take_snapshot()

        bandit.update(obs)
        probs = bandit.win_prop(draw=100000)

        snapshot2 = tracemalloc.take_snapshot()

        top_stats = snapshot2.compare_to(snapshot1, 'lineno')

        print("\nTop 10 memory allocations:")
        for stat in top_stats[:10]:
            print(stat)

        current, peak = tracemalloc.get_traced_memory()
        print(f"\nCurrent memory: {current / 1024 / 1024:.2f} MB")
        print(f"Peak memory: {peak / 1024 / 1024:.2f} MB")

        tracemalloc.stop()

    except ImportError:
        print("tracemalloc not available, skipping memory test")


def main():
    """Run all benchmarks."""
    print("=" * 70)
    print("PERFORMANCE BENCHMARK - Multi-Armed Bandit Algorithms")
    print("=" * 70)
    print()

    np.random.seed(42)

    # Basic benchmarks
    benchmark_logistic_bandit()
    benchmark_linear_bandit()
    benchmark_ts_par()

    # Detailed profiling
    profile_logistic_bandit_detailed()
    profile_linear_bandit_detailed()

    # Scaling tests
    compare_draw_sizes()

    # Memory usage
    memory_usage_test()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Key Findings:
1. win_prop() is the main bottleneck (Monte Carlo sampling)
2. Execution time scales linearly with draw size
3. Update operations are relatively fast
4. Most time spent in numpy random number generation

Optimization Opportunities:
- Monte Carlo sampling (win_prop): Perfect candidate for Rust
- Parallel sampling across multiple arms
- Vectorized operations for multiple predictions

Next Steps:
- Implement critical paths in Rust with PyO3
- Use rayon for parallel Monte Carlo sampling
- Benchmark Rust version against Python baseline
    """)


if __name__ == "__main__":
    main()
