# LogisticBandit

[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

LogisticBandit is a Python module for Multi-armed Bandit problems with logistic models. It implements both Full-Rank Thompson Sampling (Full-TS) and Odds Ratio Thompson Sampling (ORTS), as described in [this paper](https://arxiv.org/abs/2003.01905).

**ORTS is generally preferred over Full-TS or Beta-Bernoulli Thompson Sampling** because it is more robust to time-varying effects, making it ideal for real-world A/B testing scenarios where user behavior changes over time.

## Features

- ✅ **Two Thompson Sampling Methods**:
  - ORTS (Odds Ratio Thompson Sampling) - robust to time-varying effects
  - Full-TS (Full Rank Thompson Sampling) - traditional approach
  - Beta-Bernoulli Thompson Sampling - simple baseline

- ✅ **Type-Safe**: Full type hints for better IDE support and code safety

- ✅ **Input Validation**: Comprehensive validation with clear error messages

- ✅ **Well-Documented**: Detailed docstrings in NumPy style

- ✅ **Configurable**: Flexible parameters for exploration/exploitation trade-offs

- ✅ **Numerically Stable**: Robust handling of edge cases and numerical issues

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

### Basic Usage (ORTS)

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

### Beta-Bernoulli Thompson Sampling

```python
from ts import TSPar

# Traditional Beta-Bernoulli approach
ts_bandit = TSPar()
obs = {"arm_1": [1000, 50], "arm_2": [1000, 45]}
ts_bandit.update(obs)

probabilities = ts_bandit.win_prop()
```

## API Reference

### LogisticBandit

**Main class for ORTS and Full-TS algorithms.**

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

### TSPar

**Beta-Bernoulli Thompson Sampling baseline.**

#### Methods

- `update(obs)`: Update Beta distributions with observations
- `win_prop(draw=10000)`: Calculate winning probabilities
- `get_models()`: Get list of tracked arms

## Examples

See `example.py` for more usage examples.

## Mathematical Background

This implementation is based on the paper:
- **Paper**: [Odds Ratio Thompson Sampling](https://arxiv.org/abs/2003.01905)
- **Key Idea**: ORTS uses odds ratios to model arm differences, which provides robustness to time-varying effects compared to traditional approaches

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

Unit tests are planned. See the `test/` directory for simulation scripts.

## Performance Considerations

- Default Monte Carlo samples: 100,000 for LogisticBandit, 10,000 for TSPar
- Reduce `draw` parameter for faster computation (at the cost of accuracy)
- For production use with many arms, consider adjusting `draw` based on your latency requirements

## Changelog

### Version 0.1.0 (Latest)
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
