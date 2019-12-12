from logisticbandit import LogisticBandit
import numpy as np
import json


class BanditParameters(LogisticBandit):

    def update(self, obs, alpha_0 = 10., max_iter = 10000, \
        odds_ratios_only = True, remove_not_observed = False, discount = .2):
        
        for m in self.get_models():
	        assert m in obs.keys()
        
        super().update(obs, alpha_0, max_iter, odds_ratios_only, remove_not_observed, discount)

    def dump(self):
        return {
            "mu": self.mu.tolist(),  
            "sigma_inv": self.sigma_inv.tolist(), 
            "action_list": self.action_list
        }
    
    def load(self, p):
        self.__init__(np.array(p["mu"]), np.array(p["sigma_inv"]), p["action_list"])

    def test(self):
        json.dumps(self.dump())

    def delete(self, action):
        action_list_new = [a for a in self.action_list if a != action]
        self.transform(action_list_new)

    def win_prop(self, aggressive = 1.0, draw = 100000):
        return super().win_prop(aggressive, draw)

