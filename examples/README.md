# Examples

This directory contains practical examples demonstrating how to use the LogisticBandit package.

## Files

### 1. `basic_usage.py`
**Basic operations with LogisticBandit**

Demonstrates:
- Creating a LogisticBandit instance
- Updating with observations
- Getting winning probabilities
- Using ORTS vs Full-TS
- Controlling exploration/exploitation with `aggressive` parameter

```bash
python examples/basic_usage.py
```

### 2. `comparison.py`
**Comparing ORTS, Full-TS, and Beta-Bernoulli Thompson Sampling**

Demonstrates:
- Side-by-side comparison of three methods
- When to use each method
- Time-varying effects and the `decay` parameter
- ORTS robustness to changing environments

```bash
python examples/comparison.py
```

**Output:**
```
Winning Probabilities:
------------------------------------------------------------
Method               arm_1           arm_2           arm_3
------------------------------------------------------------
ORTS                 0.3421          0.4156          0.2423
Full-TS              0.3398          0.4178          0.2424
Beta-Bernoulli       0.3405          0.4172          0.2423
------------------------------------------------------------
```

### 3. `ab_testing.py`
**Realistic A/B Testing with Thompson Sampling**

Demonstrates:
- Dynamic traffic allocation in A/B tests
- Reducing regret compared to equal allocation
- Multi-variant testing (A/B/C tests)
- Realistic simulation with binomial sampling

```bash
python examples/ab_testing.py
```

**Output:**
```
FINAL RESULTS:
----------------------------------------------------------------------
Variant         Views        Conversions     Obs. Rate    True Rate
----------------------------------------------------------------------
control         31,245       1,562           5.00%        5.00%
variant_a       42,891       2,230           5.20%        5.20%
variant_b       25,864       1,241           4.80%        4.80%
----------------------------------------------------------------------
Total Users: 100,000
Regret: 127 (0.13%)

Thompson Sampling allocates more traffic to better-performing variants!
```

## Quick Start

### Example 1: Simple Two-Arm Bandit
```python
from logisticbandit import LogisticBandit

# Create bandit
bandit = LogisticBandit()

# Update with observations
obs = {"control": [10000, 500], "variant": [10000, 520]}
bandit.update(obs)

# Get winning probabilities
probs = bandit.win_prop()
print(probs)  # {'control': 0.42, 'variant': 0.58}
```

### Example 2: Time-Varying Effects
```python
from logisticbandit import LogisticBandit

bandit = LogisticBandit()

# Day 1
bandit.update({"a": [1000, 50], "b": [1000, 40]})

# Day 2 (use decay to forget old data)
bandit.update({"a": [1000, 45], "b": [1000, 55]}, decay=0.2)

probs = bandit.win_prop()
# Variant B now has higher probability due to recent performance
```

### Example 3: Aggressive Exploration/Exploitation
```python
from logisticbandit import LogisticBandit

bandit = LogisticBandit()
bandit.update({"a": [1000, 55], "b": [1000, 45]})

# More exploration (less aggressive)
probs_explore = bandit.win_prop(aggressive=0.5)
print(probs_explore)  # {'a': 0.58, 'b': 0.42}

# More exploitation (more aggressive)
probs_exploit = bandit.win_prop(aggressive=3.0)
print(probs_exploit)  # {'a': 0.73, 'b': 0.27}
```

## Running Examples

All examples are standalone Python scripts:

```bash
# Run from project root
python examples/basic_usage.py
python examples/comparison.py
python examples/ab_testing.py

# Or run from examples directory
cd examples
python basic_usage.py
python comparison.py
python ab_testing.py
```

## Common Patterns

### Pattern 1: Online Learning Loop
```python
bandit = LogisticBandit()

for round in range(num_rounds):
    # Get allocation
    if round == 0:
        allocation = equal_allocation()
    else:
        allocation = bandit.win_prop()

    # Serve users and collect data
    observations = serve_users(allocation)

    # Update model
    bandit.update(observations)
```

### Pattern 2: Multi-Armed Bandits
```python
# Many arms
bandit = LogisticBandit()
obs = {
    f"variant_{i}": [1000, int(1000 * rate)]
    for i, rate in enumerate([0.05, 0.052, 0.048, 0.051, 0.049])
}
bandit.update(obs)
probs = bandit.win_prop()
```

### Pattern 3: Comparing Methods
```python
orts = LogisticBandit()
full_ts = LogisticBandit()

orts.update(obs, odds_ratios_only=True)
full_ts.update(obs, odds_ratios_only=False)

orts_probs = orts.win_prop()
full_ts_probs = full_ts.win_prop()
```

## Tips

1. **Use ORTS (default)** for most applications - it's robust to time-varying effects
2. **Use `decay` parameter** when user behavior changes over time
3. **Use `aggressive` parameter** to control exploration vs exploitation
4. **Use `draw` parameter** to trade off accuracy vs speed
5. **Start with equal allocation** then switch to Thompson Sampling after initial data

## Further Reading

- [README.md](../README.md) - Full documentation
- [tests/](../tests/) - Unit tests with more examples
- [Paper](https://arxiv.org/abs/2003.01905) - Original ORTS paper

## Questions?

If you have questions or need help:
1. Check the [README.md](../README.md) for API documentation
2. Look at the unit tests in [tests/](../tests/)
3. Open an issue on GitHub
