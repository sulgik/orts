# Robust Multi-Armed Bandit

```
from logisticbandit import LogisticBandit 

# argument default: odds_ratos_only = True, discount = .0

orpar = LogisticBandit()


# first update

obs = {"model_1": [30000, 300], "model_2": [30000, 290]}
orpar.update(obs)

orpar.get_models()
orpar.get_par(["model_1", "model_2"])

orpar.win_prop()


# Use Odds Ratio prior only
obs = {"model_1": [20000, 200], "model_2": [20000, 180], "model_3": [20000, 210]}
orpar.update(obs)


orpar.win_prop()


# Full Rank update

obs = {"model_1": [30000, 310], "model_3": [30000, 300]}
orpar.update(obs, odds_ratios_only=False)

orpar.win_prop()

```
