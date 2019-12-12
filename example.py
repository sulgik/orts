from logisticbandit import LogisticBandit
import numpy as np
from numpy.linalg import inv

# construct an experiment
orpar = LogisticBandit()
fupar = LogisticBandit()

# first observation
obs = {"model_1": [1000, 51], "model_2": [1000, 50], "model_3": [1000, 49]}

orpar.update(obs)
fupar.update(obs, odds_ratios_only = True)

orpar.win_prop()
fupar.win_prop()

# second overvation (model insert)
obs = {"model_2": [100, 5]}
#, "model_3":[10000, 400]}

orpar.update(obs)
fupar.update(obs, odds_ratios_only = False)

orpar.win_prop()
fupar.win_prop()


# third overvation (model remove)
obs = {"model_1": [30000, 310], "model_3": [30000, 300]}

orpar.update(obs)

orpar.action_list
orpar.mu
orpar.sigma_inv

np.linalg.pinv(orpar.sigma_inv)
orpar.win_prop()

# one can control aggressive parameter in win_prop (default is 1.0)
orpar.win_prop(aggressive = 0.2)

