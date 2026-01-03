# Multi-Armed Bandit with Thompson Sampling

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python library for Multi-armed Bandit problems supporting both **binary outcomes** (LogisticBandit) and **continuous rewards** (LinearBandit).

- **LogisticBandit**: For binary outcomes (clicks, conversions, etc.) using logistic regression with Thompson Sampling. Implements Full-Rank Thompson Sampling (Full-TS) and Odds Ratio Thompson Sampling (ORTS), as described in [this paper](https://arxiv.org/abs/2003.01905).

- **LinearBandit**: For continuous outcomes (revenue, latency, ratings, etc.) using Gaussian Thompson Sampling with Bayesian updating.

**ORTS is generally preferred over Full-TS or Beta-Bernoulli Thompson Sampling** for binary outcomes because it is more robust to time-varying effects, making it ideal for real-world A/B testing scenarios where user behavior changes over time.

## Features

- ✅ **Multiple Bandit Types**:
  - **LogisticBandit** for binary outcomes (clicks, conversions)
    - ORTS (Odds Ratio Thompson Sampling) - robust to time-varying effects
    - Full-TS (Full Rank Thompson Sampling) - traditional approach
  - **LinearBandit** for continuous rewards (revenue, latency, ratings)
    - Gaussian Thompson Sampling with conjugate Bayesian updates
  - **Beta-Bernoulli Thompson Sampling** - simple baseline for binary outcomes

- ✅ **Type-Safe**: Full type hints for better IDE support and code safety

- ✅ **Input Validation**: Comprehensive validation with clear error messages

- ✅ **Well-Documented**: Detailed docstrings in NumPy style

- ✅ **Configurable**: Flexible parameters for exploration/exploitation trade-offs

- ✅ **Numerically Stable**: Robust handling of edge cases and numerical issues

- ✅ **Well-Tested**: Comprehensive test suite with 100+ unit tests and 88%+ coverage

## Installation

### From source
```bash
# Clone the repository
git clone https://github.com/sulgik/orts.git
cd orts

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Development installation
```bash
# Install with development dependencies
pip install -e ".[dev]"
```

## Quick Start

### LogisticBandit - For Binary Outcomes

Use LogisticBandit when your outcomes are binary (success/failure, click/no-click, convert/not-convert).

#### Basic Usage (ORTS)

```python
from logisticbandit import LogisticBandit

# Create a LogisticBandit instance (ORTS by default)
bandit = LogisticBandit()

# First observation: arm_1 had 300 successes out of 30000 trials
obs = {"arm_1": [30000, 300], "arm_2": [30000, 290]}
bandit.update(obs)

# Get winning probabilities for each arm
probabilities = bandit.win_prop()
print(probabilities)
# Output: {'arm_1': 0.543, 'arm_2': 0.457}

# Get the list of tracked arms
print(bandit.get_models())
# Output: ['arm_1', 'arm_2']
```

### Adding New Arms

```python
# Add a new arm (arm_3) to the existing model
obs = {"arm_1": [20000, 200], "arm_2": [20000, 180], "arm_3": [20000, 210]}
bandit.update(obs)

probabilities = bandit.win_prop()
print(probabilities)
# Output: {'arm_1': 0.312, 'arm_2': 0.287, 'arm_3': 0.401}
```

### Full-Rank Thompson Sampling

```python
# Use Full-TS instead of ORTS
obs = {"arm_1": [30000, 310], "arm_3": [30000, 300]}
bandit.update(obs, odds_ratios_only=False)

probabilities = bandit.win_prop()
```

### Advanced Options

```python
# Adjust exploration/exploitation trade-off
probabilities = bandit.win_prop(aggressive=2.0)  # More exploitation
probabilities = bandit.win_prop(aggressive=0.5)  # More exploration

# Use decay to discount old observations (useful for non-stationary environments)
bandit.update(obs, decay=0.1)  # 10% discount on prior information

# Adjust sampling precision
probabilities = bandit.win_prop(draw=50000)  # More samples = more accurate
```

#### Beta-Bernoulli Thompson Sampling

```python
from ts import TSPar

# Traditional Beta-Bernoulli approach
ts_bandit = TSPar()
obs = {"arm_1": [1000, 50], "arm_2": [1000, 45]}
ts_bandit.update(obs)

probabilities = ts_bandit.win_prop()
```

### LinearBandit - For Continuous Rewards

Use LinearBandit when your outcomes are continuous values (revenue, response time, ratings, etc.).

#### Basic Usage

```python
from linearbandit import LinearBandit

# Create a LinearBandit instance
bandit = LinearBandit(obs_noise=1.0)

# Observations are continuous reward values
obs = {
    "variant_a": [2.3, 2.7, 2.4, 2.6, 2.5],  # Multiple observations
    "variant_b": [3.1, 3.3, 3.0],             # Different number is fine
    "variant_c": 2.8                          # Single value also works
}
bandit.update(obs)

# Get winning probabilities
probabilities = bandit.win_prop()
print(probabilities)
# Output: {'variant_b': 0.65, 'variant_c': 0.25, 'variant_a': 0.10}
```

#### Advanced Options

```python
# Get detailed statistics
stats = bandit.get_statistics()
for action, stat in stats.items():
    print(f"{action}: mean={stat['mu']:.3f}, uncertainty={stat['sigma']:.3f}")

# Adjust exploration/exploitation
probabilities = bandit.win_prop(aggressive=2.0)  # More exploitation

# Use decay for non-stationary environments
bandit.update(obs, decay=0.2)  # Discount old information

# Custom noise parameters
bandit = LinearBandit(obs_noise=0.5, default_sigma=2.0)
```

## API Reference

### LogisticBandit

**For binary outcomes using ORTS and Full-TS algorithms.**

#### Methods

- `update(obs, odds_ratios_only=True, decay=0.0)`: Update model with new observations
  - `obs`: Dictionary mapping arm names to `[total_count, success_count]`
  - `odds_ratios_only`: If True, use ORTS; if False, use Full-TS
  - `decay`: Discount factor for prior information (0.0 to 1.0)

- `win_prop(action_list=None, draw=100000, aggressive=1.0)`: Calculate winning probabilities
  - `action_list`: List of arms to consider (None = all arms)
  - `draw`: Number of Monte Carlo samples
  - `aggressive`: Exploration/exploitation parameter

- `get_models()`: Get list of currently tracked arms

- `get_par(action_list)`: Get transformed parameters for specific arms

### LinearBandit

**For continuous rewards using Gaussian Thompson Sampling.**

#### Methods

- `update(obs, decay=0.0)`: Update model with new observations
  - `obs`: Dictionary mapping action names to rewards (list of floats or single float)
  - `decay`: Discount factor for prior information (0.0 to 1.0)

- `win_prop(action_list=None, draw=100000, aggressive=1.0)`: Calculate winning probabilities
  - `action_list`: List of actions to consider (None = all actions)
  - `draw`: Number of Monte Carlo samples
  - `aggressive`: Exploration/exploitation parameter

- `get_models()`: Get list of currently tracked actions

- `get_statistics()`: Get detailed statistics (mu, sigma, count, mean_reward) for all actions

#### Constructor Parameters

- `mu`: Prior mean for each action (dict)
- `sigma`: Prior standard deviation for each action (dict)
- `default_sigma`: Default prior uncertainty for new actions (default: 1.0)
- `obs_noise`: Assumed observation noise standard deviation (default: 1.0)

### TSPar

**Beta-Bernoulli Thompson Sampling baseline for binary outcomes.**

#### Methods

- `update(obs)`: Update Beta distributions with observations
- `win_prop(draw=10000)`: Calculate winning probabilities
- `get_models()`: Get list of tracked arms

## Examples and Tutorials

### 📚 New to Multi-Armed Bandits?

Start with the comprehensive tutorial:

- **`TUTORIAL.md`** - Complete guide to Multi-Armed Bandits and Thompson Sampling
  - Why use bandits? (Exploration-Exploitation tradeoff)
  - When to use LogisticBandit vs LinearBandit
  - Real-world examples with detailed explanations
  - Common mistakes and how to avoid them

- **`examples/tutorial_step_by_step.py`** - Interactive step-by-step tutorial
  - Run it to learn concepts interactively
  - Covers basic usage, parameters, and best practices

```bash
# Start with the interactive tutorial
python examples/tutorial_step_by_step.py

# Read the comprehensive guide
cat TUTORIAL.md
```

### 🚀 Ready to Use?

Practical examples are available in the `examples/` directory:

- **`examples/basic_usage.py`** - LogisticBandit basics
- **`examples/comparison.py`** - Compare ORTS, Full-TS, and Beta-Bernoulli
- **`examples/ab_testing.py`** - Realistic A/B testing simulation
- **`examples/linear_bandit.py`** - LinearBandit for continuous rewards

Run any example:
```bash
python examples/linear_bandit.py
python examples/comparison.py
```

## Mathematical Background

### LogisticBandit (ORTS)

This implementation is based on the paper:
- **Paper**: [Odds Ratio Thompson Sampling](https://arxiv.org/abs/2003.01905)
- **Key Idea**: ORTS uses odds ratios to model arm differences, which provides robustness to time-varying effects compared to traditional approaches

### LinearBandit (Gaussian Thompson Sampling)

Uses Bayesian inference with Gaussian conjugate priors:
- **Prior**: Each action's mean reward follows N(μ₀, σ₀²)
- **Likelihood**: Observations are N(μ, σ²) where σ is the observation noise
- **Posterior**: Also Gaussian (conjugate prior property)
- **Thompson Sampling**: Sample from posterior and select action with highest sample

## Contributing

Contributions are welcome! For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests (once implemented)
pytest

# Check types (recommended)
mypy logisticbandit.py ts.py utils.py
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_linearbandit.py
pytest tests/test_logisticbandit.py
```

Current test coverage: **88%+** with 100+ unit tests

## Performance Considerations

- Default Monte Carlo samples: 100,000 for LogisticBandit, 10,000 for TSPar
- Reduce `draw` parameter for faster computation (at the cost of accuracy)
- For production use with many arms, consider adjusting `draw` based on your latency requirements

## Changelog

### Version 0.2.0 (Latest)
- ✅ **NEW**: Added LinearBandit for continuous rewards (Gaussian Thompson Sampling)
- ✅ **NEW**: Comprehensive test suite with 100+ tests (88%+ coverage)
- ✅ **NEW**: Multiple examples in `examples/` directory
- ✅ **NEW**: CI/CD with GitHub Actions
- ✅ Type checking with mypy
- ✅ Code coverage analysis

### Version 0.1.0
- ✅ Fixed critical `exit()` bug that caused program termination
- ✅ Added comprehensive docstrings
- ✅ Added type hints for all public methods
- ✅ Added input validation with clear error messages
- ✅ Made hardcoded values configurable
- ✅ Added `requirements.txt` and `setup.py` for easy installation
- ✅ Improved numerical stability in covariance matrix handling

## Author

* **Sulgi Kim** - [GitHub](https://github.com/sulgik)

## License

[MIT](https://choosealicense.com/licenses/mit/)

## Citation

If you use this code in your research, please cite the original paper:

```bibtex
@article{orts2020,
  title={Odds Ratio Thompson Sampling},
  author={Kim, Sulgi},
  journal={arXiv preprint arXiv:2003.01905},
  year={2020}
}
```
