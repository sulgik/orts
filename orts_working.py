import numpy as np
# import matplotlib.pyplot as plt
# from __future__ import division
import math
from numpy.linalg import inv
    
class ORPar(object):
    # OR paramter class which does not depend on reference
    def __init__(self):
        # build from npq
        # orpar has action_list and action_ref
        self.action_ref = None
        self.mu = None
        
    def reset(self, mu, Sigma_inv, action_nonref, action_ref):
        self.mu = mu
        self.Sigma_inv = Sigma_inv
        self.action_nonref = action_nonref
        self.action_ref = action_ref
        self.Sigma = inv(Sigma_inv)

    def get_par(self, action_nonref, action_ref):
        # assume action_nonref and action_ref is same
        transform_mat = np.zeros((len(self.action_nonref), len(action_nonref)))

        if action_ref != self.action_ref:
            transform_mat[:, self.action_ref.index(action_nonref)] = -1.0

        for i, new_nonref in enumerate(action_nonref):        
            for j, old_nonref in enumerate(self.action_nonref):
                if new_nonref == old_nonref:
                    transform_mat[i,j] = 1.0

        mu = np.matmul(transform_mat, self.mu)        
        Sigma = transform_mat.dot(self.Sigma).dot(np.transpose(transform_mat))
        Sigma_inv = inv(self.Sigma)
        
        return mu, Sigma, Sigma_inv

    def transform(self, action_nonref, action_ref):
        pars = self.get_par(action_nonref, action_ref)
        mu = pars[0]
        new_obj = ORPar().reset(self.mu, self.Sigma_inv, action_nonref, action_ref)
        return new_obj

    def update(self, obs, alpha_0 = .01, max_iter = 100):
        # estimate       
        # mu, Sigma_inv, action_nonref, action_ref = \
        estimates = estimate(self, obs, alpha_0 = alpha_0, max_iter = max_iter)
        mu = estimates[0]
        Sigma_inv = estimates[1]
        action_nonref = estimates[2]
        action_ref = estimates[3]
        
        # constuct orpar object
        
        self.mu = mu
        self.Sigma_inv = Sigma_inv
        self.Sigma = inv(Sigma_inv)
        self.action_nonref = action_nonref 
        self.action_ref = action_ref

def build_fns(par, obs, action_newcome, action_on):
    # target_fn constructor

    total_number = 0.0
    for value in obs.values():
        total_number += float(value[0])

    action_list = action_newcome + action_on

    total_cnt = np.empty(0)
    success_cnt = np.empty(0)
    for action in action_list:
        total_cnt = np.append(total_cnt, float(obs[action][0]))
        success_cnt = np.append(success_cnt, float(obs[action][1]))
        
    def posterior(w):
        # need to organize arguments
        # w : list
        # first we assume reference is in prior_for_in_par
        
        p = [logistic(w[i] + w[-1]) for i in range(len(obs) - 1)]
        p.append(logistic(w[-1]))

        # prior
        if len(action_on) < 2:
            prior_log = 0
        else:
            w_mu_diff = np.array(w[-len(action_on):-1]) - par.mu
            prior_log = \
                -0.5 * w_mu_diff.dot(par.Sigma_inv).dot(np.transpose(w_mu_diff)) / \
                    total_number

        # likelihood
        likelihood_log = 0.0
        for i, action in enumerate(action_list):
            likelihood_log += \
                (float(obs[action][1]) * math.log(p[i]) + float(obs[action][0] - obs[action][1]) * math.log(1 - p[i])) / \
                        total_number

        posterior_log = prior_log + likelihood_log
        return posterior_log

    def gradient(w):
        # gradient for K 
        # w : list
        p = np.array([logistic(w[i] + w[-1]) for i in range(len(obs) - 1)])
        p = np.append(p, logistic(w[-1]))

        # prior part
        gradient = np.zeros(len(obs))
        if len(action_on) >= 2:
            gradient[-len(action_on):-1] = np.matmul(-np.array(w[-len(action_on):-1]) - par.mu, par.Sigma_inv) / total_number

        # likelihood part
        gradient += (success_cnt - p * total_cnt) / total_number
        return gradient

    return posterior, gradient
        
def monte_carlo(orpar, action_list = None, draw = 1000):
    # construct prior for action_list (mu and sigma)
    
    orpar_action_list = orpar.action_nonref + [orpar.action_ref]
    if action_list is None:
        action_list = orpar_action_list

    action_on = []
    action_on_index = []
    action_newcome = []

    for action_input in action_list:    
        if action_input in orpar.action_nonref + [orpar.action_ref]:                
            action_on.append(action_input)
            # actoin_on_index.append(i)
        else: 
            action_newcome.append(action_input)

    n = len(action_list)
    mu = np.zeros(n-1)
    Sigma = np.full((n-1,n-1), 1000.0)
    
    if len(action_on) >= 2:
        mu_0, Sigma_0, Sigma_inv_0 = orpar.get_par(action_on[:-1], action_on[-1])
        if Sigma_0 is None:
            Sigma_0 = inv(Sigma_inv_0)
        # place 
        n2 = len(action_on)
        mu[-(n2-1):] = mu_0
        Sigma[-(n2-1):, -(n2-1):] = Sigma_0
        action_list_temp = action_on[:-1] + action_newcome + [action_on[-1]]
    else:
        action_list_temp = action_list

    # Generation
    mc = np.random.multivariate_normal(mu, Sigma, draw)
    mc = np.concatenate((mc, np.zeros([draw, 1])), axis = 1)

    # count frequency of each arm being winner 
    counts = [0.0 for _ in range(len(mu)+1)]
    winner_idxs = np.asarray(mc.argmax(axis = 1)).reshape(draw, )
    for idx in winner_idxs:
        counts[idx] += 1.0
    
    # divide by draw to approximate probability distribution
    p_winner_temp = [count / draw for count in counts]
    
    # sort
    p_winner = {}
    for _, action in enumerate(action_list):
        for i, action_temp in enumerate(action_list_temp): 
            if action == action_temp:
                p_winner[action] = p_winner_temp[i]
    
    return p_winner


def estimate_mu(target_fn, gradient_fn, initial, alpha_0 = 0.01, max_iter = 100):
    # general estimate function
    # return mu MAP ordered by obs_not_in_par and obs_not_par

    # initial value
    w = initial

    max_w, max_value = None, float("-Inf")
    iterations_with_no_improvement = 0
        
    # w to 
    while iterations_with_no_improvement < max_iter:
        value = target_fn(w)
        
        if value > max_value:
            max_w, max_value = w, value
            iterations_with_no_improvement = 0
            alpha = alpha_0
            
        else:
            iterations_with_no_improvement += 1
            alpha *= .9
        
        gradient = gradient_fn(w)
        w = vector_add(w, scalar_multiply(alpha, gradient))

    return max_w

def estimate(par, obs, alpha_0 = .01, max_iter = 10000):
    # combine par and obs
    obs_actions = obs.keys()
    par_actions = par.action_nonref + [par.action_ref]
    total_number = 0.0
    for value in obs.values():
        total_number += value[0]

    action_on = []
    # action_on_index = []
    action_newcome = []

    for action_obs in obs_actions:
        # for i, action_par in enumerate(par_actions):    
        if action_obs in par.action_nonref + [par.action_ref]:                
            action_on.append(action_obs)
            # actoin_on_index.append(i)
        else: 
            action_newcome.append(action_obs)

    action_list = action_newcome + action_on

    # mu
    par.transform(action_on[:-1], action_on[-1])
    target_fn, gradient_fn = build_fns(par, obs, action_newcome, action_on)

    initial = np.concatenate((par.mu, np.repeat(0.0, len(action_newcome)), [-1.0]))
    mu_list = estimate_mu(target_fn, gradient_fn, initial = initial, alpha_0 = alpha_0, max_iter = max_iter)
    
    # Sigma_inv
    n = len(obs)
    Sigma_inv_prior = np.zeros((n-1, n-1))
    if len(action_on) >= 2:
        Sigma_inv_prior[-(len(action_on)-1):, -(len(action_on)-1):] = par.Sigma_inv

    p = np.empty(0)
    for mu in mu_list[:-1]:
        p = np.append(p, logistic(mu + mu_list[-1]))
    p_ref = logistic(mu_list[-1])

    total_cnt = np.empty(0)
    for action in action_list[:-1]:
        total_cnt = np.append(total_cnt, float(obs[action][0]))
    total_cnt_ref = obs[action_list[-1]][0]

    ref_npq = total_number / (total_cnt_ref * p_ref * (1 - p_ref)) 
    Sigma_inv_likelihood = np.full((n-1, n-1), ref_npq)
    diagonal_array = total_number / (total_cnt * p * (1 - p)) + ref_npq
    for i, diag_ii in enumerate(diagonal_array):
        Sigma_inv_likelihood[i, i] = diag_ii
    Sigma_inv = Sigma_inv_prior + Sigma_inv_likelihood

    return mu_list[:-1], Sigma_inv, action_list[:-1], action_list[-1]

def mu_Sigmainv(obs):
    action_list = [i for i in obs.keys()]
    par = ORPar()
    target_fn, gradient_fn = build_fns([], obs, action_list)
    initial = np.repeat(0.0, len(obs))
    mu_list = estimate_mu(target_fn, gradeint_fn, initial = initial, alpha_0 = alpha_0, max_iter = max_iter)
    
    # Sigma_inv
    n = len(obs)
    Sigma_inv_prior = np.zeros((n-1, n-1))
    Sigma_inv_prior[-(len(action_on)-1):, -(len(action_on)-1):] = par_sub.Sigma_inv

    p = np.empty(0)
    for mu in mu_list[:-1]:
        p.append(logistic(mu + mu[-1]))
    p_ref = logistic(mu[-1])

    total_cnt = np.empty(0)
    for action in action_list:
        total_cnt.append(float(obs[action][0]))

    Sigma_inv_likelihood = np.diag(1.0 / (total_cnt * p * (1 - p))) + np.fill(1.0 / (n * p_ref * (1 - p_ref), (n-1, n-1)))
    Sigma_inv = Sigma_inv_prior + Sigma_inv_likelihood

    return mu, Sigma_inv

def logistic(x):
#    print("logistic x ", x)
    return 1.0 / (1.0 + math.exp(-x))

def vector_add(v, w):
    """subtracts two vectors componentwise"""
    return [v_i + w_i for v_i, w_i in zip(v,w)]

def vector_subtract(v, w):
    """subtracts two vectors componentwise"""
    return [v_i + w_i for v_i, w_i in zip(v,w)]

def scalar_multiply(c, v):
    return [c * v_i for v_i in v]
    
