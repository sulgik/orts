# Robust Multi-Armed Bandit

```
from logisticbandit import LogisticBandit 

# default parameter, odds_ratos_only = True, discount = .2

orpar = LogisticBandit()


# first observation
obs = {"model_1": [30000, 300], "model_2": [30000, 290]}
orpar.update(obs)

orpar.get_models()
orpar.get_par(["model_1", "model_2"])

orpar.win_prop()


# second overvation (model insert)
obs = {"model_1": [20000, 200], "model_2": [20000, 180], "model_3": [20000, 210]}
orpar.update(obs)

orpar.win_prop()


# third overvation (model remove)
orpar.delete("model_2")

obs = {"model_1": [30000, 310], "model_3": [30000, 300]}
orpar.update(obs)


orpar.win_prop()

# one can control aggressive parameter in win_prop
orpar.win_prop(aggressive = 1.0)
```
