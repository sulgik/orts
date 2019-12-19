import numpy as np
from numpy import fill_diagonal
from numpy.linalg import pinv, inv

from utils import estimate, is_pos_semidef


class LogisticBandit(object):
    def __init__(self, mu=None, sigma_inv=None, action_list=None):
        self.mu, self.sigma_inv, self.action_list = None, None, None
        self._initialize(mu, sigma_inv, action_list)

    def _initialize(self, mu, sigma_inv, action_list):
        self.mu = np.copy(mu) if mu igis not None else np.array([])
        self.sigma_inv = np.copy(sigma_inv) if sigma_inv is not None else np.empty((0, 0))
        self.action_list = action_list.copy() if action_list is not None else []

    def get_models(self):
        return self.action_list

    def get_par(self, action_list=None):

        if not action_list:
            return None, None

        action_nonref = action_list[:-1]
        action_ref = action_list[-1]

        # transform_mat1: to new odds ratio    
        transform_mat1 = np.zeros((len(action_list), len(self.action_list)))
        transform_mat1[:, self.action_list.index(action_ref)]=-1
        for i, new_nonref in enumerate(action_nonref):
            for j, old_nonref in enumerate(self.action_list):
                if new_nonref == old_nonref:
                    transform_mat1[i,j]=1

        transform_mat1[-1, self.action_list.index(action_ref)]=1
        
        # transform_mat2: inverse from old odds ratios
        transform_mat2 = np.zeros((len(self.action_list), len(self.action_list)))
        transform_mat2[:,-1] = 1
        fill_diagonal(transform_mat2, 1)

        transform_mat = transform_mat1.dot(transform_mat2)

        mu = np.matmul(transform_mat, self.mu)

        tr_inv = np.rint(pinv(transform_mat))
        sigma_inv = tr_inv.T.dot(self.sigma_inv).dot(tr_inv)

        return mu, sigma_inv

    def transform(self, action_list):
        mu, sigma_inv = self.get_par(action_list)
        self.__init__(
            mu = mu, sigma_inv = sigma_inv, action_list = action_list)

    def update(
        self, obs, odds_ratios_only = True, remove_not_observed = False,
        discount = .0):

        obs_valid = {i:obs[i] for i in obs.keys() if obs[i][0] > 0}

        if len(obs_valid) > 0:

            total_number=0.
            for value in obs_valid.values():
                total_number += value[0]

            # find arrangement
            action_on=[]
            action_newcome=[]

            for action_obs in obs_valid.keys():
                if action_obs in self.get_models():                
                    action_on.append(action_obs)
                else: 
                    action_newcome.append(action_obs)

            action_nonobserved = [x for x in self.get_models() if x not in obs_valid.keys()]
            action_list = action_newcome + action_nonobserved + action_on

            prior = self.get_par(action_nonobserved + action_on)
            if (prior[0] is not None) and odds_ratios_only:
                if len(prior[0]) > 1:
                    sigma_inv_new = np.zeros((len(prior[0]), len(prior[0])))
                    sigma_inv_new[:-1,:-1] = inv(inv(prior[1])[:-1,:-1])
                    prior = (prior[0], sigma_inv_new)

            obs_list = [obs_valid[action] for action in action_newcome + action_on]
            index = [len(action_newcome), len(action_newcome + action_nonobserved), len(action_list)]

            # estimate part    
            if len(action_on) <= 1:
                parameters = \
                    estimate(prior, obs_list, index, discount = discount)
            
            else:
                parameters = \
                    estimate(prior, obs_list, index, discount = discount)
                
            self.mu = parameters[0]
            self.sigma_inv = parameters[1]
            self.action_list = action_list

    def win_prop(self, aggressive = 1., draw = 100000):

        if len(self.action_list) == 0:
            return {}
        elif len(self.action_list) == 1:
            return {self.action_list[0]: 1.}

        action_list = self.action_list
        
        mu = self.mu
        sigma = pinv(self.sigma_inv)
        
        # Generation depending on dimension
        if len(sigma) == 1:
            mc = np.random.normal(mu[0], np.sqrt(sigma[0]), draw)
            mc = mc.reshape(draw,1)
        elif len(sigma) == 2:
            mc = np.random.normal(mu[0], np.sqrt(sigma[0,0]), draw)
            mc = mc.reshape(draw,1)
        else:
            while not is_pos_semidef(sigma[:-1,:-1]):
                # critical case
                print("Warning: not positive semidefinite")
                exit() # should be removed
                np.fill_diagonal(sigma, sigma.diagonal() + .001)

            mc = np.random.multivariate_normal(mu[:-1], sigma[:-1,:-1], draw)

        # concatenate
        mc = np.concatenate((mc, np.zeros([draw, 1])), axis = 1)

        # count frequency of each arm being winner 
        counts = [0.0 for _ in range(len(mu))]
        winner_idxs = np.asarray(mc.argmax(axis = 1)).reshape(draw, )
        for idx in winner_idxs:
            counts[idx] += 1.
        
        # divide by draw to approximate probability distribution
        count_gamma = np.array([count**aggressive for count in counts])
        p_winner = count_gamma / np.sum(count_gamma)

        out = {}
        for i, action in enumerate(action_list):
            out[action] = p_winner[i]

        return out
