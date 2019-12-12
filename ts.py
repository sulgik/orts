import numpy as np

class TSPar(object):

    def __init__(self):
        self.action_list = []
        self.alpha = np.array([])
        self.beta = np.array([])

    def get_models(self):
        return self.action_list

    def update(self, obs, alpha_0 = .01, max_iter = 100):
        if len(self.action_list) == 0:
            action_list = [i for i in obs.keys()]
            self.action_list = action_list
            self.alpha = np.array([1]*len(action_list))
            self.beta  = np.array([1]*len(action_list))

        views  = np.array([obs[action][0] for action in self.action_list])
        clicks = np.array([obs[action][1] for action in self.action_list])

        self.alpha = self.alpha + clicks
        self.beta  = self.beta + views - clicks

    def win_prop(self, draw = 10000):

        mc = np.matrix(np.random.beta(self.alpha, self.beta, size=[draw, len(self.action_list)]))

        # count frequency of each arm being winner 
        counts = [0.0 for _ in range(len(self.action_list))]
        winner_idxs = np.asarray(mc.argmax(axis = 1)).reshape(draw, )
        for idx in winner_idxs:
            counts[idx] += 1.0
        
        # divide by draw to approximate probability distribution
        p_winner = [count / float(draw) for count in counts]

        out = {self.action_list[i]: p_winner[i] for i in range(len(p_winner))}
        return out
    
