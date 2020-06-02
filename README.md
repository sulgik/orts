# LogisticBandit
LogisticBandit is a python module for Multi-armed bandit with logistic model. 
It includes both full rank thompson sampling (Full-TS) and Odds Ratio Thompson Sampling (ORTS), which is described in [the paper](https://arxiv.org/abs/2003.01905).
In general, ORTS is desirable than Full-TS or Beta-Bernoulli Thompson sampling, because ORTS is robust to time-varying effect.

### Usage
```
from logisticbandit import LogisticBandit 

# argument default: odds_ratos_only = True, discount = .0

orpar = LogisticBandit()


# first update

obs = {"arm_1": [30000, 300], "arm_2": [30000, 290]}
orpar.update(obs)

orpar.get_models()
orpar.get_par(["arm_1", "arm_2"])

orpar.win_prop()


# Use Odds Ratio prior only
obs = {"arm_1": [20000, 200], "arm_2": [20000, 180], "arm_3": [20000, 210]}
orpar.update(obs)


orpar.win_prop()


# Full Rank update

obs = {"arm_1": [30000, 310], "arm_3": [30000, 300]}
orpar.update(obs, odds_ratios_only=False)

orpar.win_prop()
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
## Author
* Sulgi Kim
## License
[MIT](https://choosealicense.com/licenses/mit/)
