import numpy as np
from numpy.linalg import pinv
import json
    
class LogisticBandit(object):
    def __init__(self):
        self.mu = np.array([])
        self.sigma_inv = np.empty((0,0))
        self.action_list = []

    def get_models(self, odds_ratios_only = True):
        return self.action_list

    def get_par(self, action_list = []):
        if len(action_list) == 0:
            action_list = self.get_models()

        action_nonref = action_list[:-1]
        action_ref = action_list[-1]
        transform_mat = np.zeros((len(action_list), len(self.action_list)))

        if action_ref != self.get_models()[-1]:
           transform_mat[:, self.action_list[:-1].index(action_ref)] = -1.

        for i, new_nonref in enumerate(action_nonref):
            for j, old_nonref in enumerate(self.action_list[:-1]):
                if new_nonref == old_nonref:
                    transform_mat[i,j] = 1.

        transform_mat[-1, self.action_list.index(action_ref)] = 1.
        transform_mat[-1,-1] = 1.

        mu = np.matmul(transform_mat, self.mu)

        sigma = pinv(self.sigma_inv)
        sigma_new = transform_mat.dot(sigma).dot(np.transpose(transform_mat))
        sigma_inv = pinv(sigma_new)

        return mu, sigma_inv

    def transform(self, action_list):
        mu, sigma_inv = self.get_par(action_list)
        
        self.mu = mu
        self.action_list = action_list
        self.sigma_inv = sigma_inv

    def transform2(self, action_list):
        mu, sigma_inv = self.get_par(action_list)
        new_obj = LogisticBandit()
        new_obj = new_obj.reset2(mu = mu, sigma_inv = sigma_inv, action_list = action_list)
        return new_obj

    def update(self, obs, alpha_0 = .3, max_iter = 20, odds_ratios_only = True, remove_not_observed = False, discount = .0):

        obs_valid = {i:obs[i] for i in obs.keys() if obs[i][0] > 0}

        if len(obs_valid) > 1:
            total_number = 0.
            for value in obs_valid.values():
                total_number += value[0]

            action_on = []
            action_newcome = []

            for action_obs in obs_valid.keys():
                if action_obs in self.get_models():                
                    action_on.append(action_obs)
                else: 
                    action_newcome.append(action_obs)

            action_non_observed = [x for x in self.get_models() if x not in obs_valid.keys()]
            
            # estimate part    
            if len(action_on) == 0:
                par_sub = LogisticBandit()
                parameters, action_list = \
                    estimate(par_sub, obs_valid, \
                        alpha_0 = alpha_0, max_iter = max_iter, odds_ratios_only = False, discount = discount)
            
            else:
                par_sub = self.transform2(action_on)

                parameters, action_list = \
                    estimate(par_sub, obs, \
                        alpha_0 = alpha_0, max_iter = max_iter, odds_ratios_only = odds_ratios_only, discount = discount)

            par_out = LogisticBandit()
            par_out.reset(parameters[0], parameters[1], action_list)

            # combine with nonobserved part
            # 동일 ref 가 있는 경우만 생각
            if ~ remove_not_observed and len(action_non_observed) > 0 and len(action_on) > 1: #
                par_transformed = self.transform2(action_non_observed + [action_list[-1]])
                par_out = extend(par_transformed, par_out)

            self.mu = par_out.mu
            self.sigma_inv = par_out.sigma_inv
            self.action_list = par_out.get_models()

    def win_prop(self, aggressive = .1, draw = 100000):
        if len(self.action_list) == 0:
            return {}
        elif len(self.action_list) == 1:
            return {self.action_list[0]: 1.}

        action_list = self.action_list
        
        mu = self.mu
        sigma = pinv(self.sigma_inv)
        
        # Generation depending on dimension
        if len(sigma) == 2:
            mc = np.random.normal(mu[0], np.sqrt(sigma[0,0]), draw)
            mc = mc.reshape(draw,1)
        else:
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

    def reset(self, mu, sigma_inv, action_list):
        self.mu = np.array(mu)
        self.sigma_inv = np.array(sigma_inv)
        self.action_list = action_list

    def reset2(self, mu, sigma_inv, action_list):
        out = LogisticBandit()
        out.mu = mu
        out.sigma_inv = sigma_inv
        out.action_list = action_list
        return out        

def preprocess_obs(i):
    out = i
    if i[1] >= i[0]:
        out[0] = i[1] + 1
    if i[0] == 0:
        out[0] = 1
        out[1] = 0

    return out

class BanditParameters(LogisticBandit):

    def dump(self):
        return {
            "mu": self.mu.tolist(),  
            "sigma_inv": self.sigma_inv.tolist(), 
            "action_list": self.action_list
        }
    
    def load(self, p):
        self.reset(p["mu"], p["sigma_inv"], p["action_list"])

    def get_models(self, odds_ratios_only = True):
        return self.action_list

    def test(self):
        json.dumps(self.dump())

    def delete(self, action):
        action_list_new = [a for a in self.action_list if a != action]
        self.transform(action_list_new)



def estimate(par, obs, alpha_0, max_iter, odds_ratios_only, discount = 0.):
    obs_actions = [i for i in obs.keys()]

    total_number = 0.
    for value in obs.values():
        total_number += value[0]

    action_on = []
    action_newcome = []

    for action_obs in obs_actions:
        if action_obs in par.get_models():                
            action_on.append(action_obs)
        else: 
            action_newcome.append(action_obs)

    actions_update = action_newcome + action_on

    # if (odds_ratios_only and len(action_on) <= 1) or (~odds_ratios_only and len(action_on) == 0):
    if len(action_on) == 0:
        # prior does not apply
        # action_newcome = obs_actions
        # action_on = []
        action_list = action_newcome + action_on
    else:
        action_list = action_on[:-1] + action_newcome + [action_on[-1]]

    # mu
    if len(action_on) == 0:
        par_sub = LogisticBandit() 
    else:    
        par_sub = par
        par_sub.transform(action_on)
    
    target_fn, gradient_fn = _build_fns(par_sub, obs, action_newcome, action_on, odds_ratios_only, discount)

    # initial value for estimate
    p_ref = float(obs[actions_update[-1]][1]) / float(obs[actions_update[-1]][0])
    if p_ref == 1.:
        p_ref = .9999
    elif p_ref == .0:
        p_ref = .0001

    if odds_ratios_only:
        if len(action_on) >= 1:
            initial = np.concatenate((
                np.repeat(0.0, len(actions_update) - len(par_sub.mu)),
                par_sub.mu[:-1], 
                [np.log(p_ref / (1. - p_ref))]))
        else:
            initial = np.concatenate((
                np.repeat(0., len(actions_update) - 1),
                [np.log(p_ref / (1. - p_ref))]))
    else:
        initial = np.concatenate((
            np.repeat(0., len(actions_update) - len(par_sub.mu)),
            par_sub.mu))

    mu_hat = \
        estimate_mu(target_fn, gradient_fn, \
            initial = initial, alpha_0 = alpha_0, max_iter = max_iter)

    # sigma_inv

    # sigma_inv prior part
    n = len(action_list)
    sigma_inv_prior = np.zeros((n, n))

    if len(action_on) > 0:
        sigma_inv_prior[:(len(action_on)-1),:(len(action_on)-1)] = par_sub.sigma_inv[:-1,:-1]

        if ~odds_ratios_only:
            sigma_inv_prior[:(len(action_on)-1),-1] = par_sub.sigma_inv[:-1,-1]
            sigma_inv_prior[-1,:(len(action_on)-1)] = par_sub.sigma_inv[-1,:-1]
            sigma_inv_prior[-1,-1] = par_sub.sigma_inv[-1,-1]

    # sigma_inv likelihood part
    p = np.array([logistic(mu + mu_hat[-1]) for mu in mu_hat[:-1]])
    p_ref = logistic(mu_hat[-1])

    total_cnt = np.array([float(obs[action][0]) for action in actions_update[:-1]])
    total_cnt_ref = float(obs[actions_update[-1]][0])

    npq = total_cnt * p * (1. - p)
    npq_ref = total_cnt_ref * p_ref * (1. - p_ref)
    sigma_inv_likelihood_full = np.zeros((n, n))

    for i in range(n-1):
        sigma_inv_likelihood_full[i, i] = npq[i]
        sigma_inv_likelihood_full[i, -1] = npq[i]
        sigma_inv_likelihood_full[-1, i] = npq[i]
    
    sigma_inv_likelihood_full[-1, -1] = np.sum(npq) + npq_ref

    # add
    if odds_ratios_only:
        sigma_likelihood_full = pinv(sigma_inv_likelihood_full)
        sigma_inv_likelihood = pinv(sigma_likelihood_full[:-1,:-1])        
        sigma_inv = (1. - discount) * sigma_inv_prior
        sigma_inv[:-1,:-1] += sigma_inv_likelihood

    else:
        sigma_inv = (1. - discount) * sigma_inv_prior + sigma_inv_likelihood_full
        
    return [mu_hat, sigma_inv], action_list



def estimate_mu(target_fn, gradient_fn, initial, alpha_0, max_iter):

    w = initial

    max_w, max_value = None, float("-Inf")
    iterations_with_no_improvement = 0
        
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

        w += alpha * gradient

    return max_w


def logistic(x):
    if x > 0:
        return 1. / (1. + np.exp(-x))
    else:
        return np.exp(x) / (1. + np.exp(x))

def _build_fns(par, obs, action_newcome, action_on, odds_ratios_only, discount):
    #
    # target_fn, gradient_fn constructor
    #
    total_number = 0.0
    for value in obs.values():
        total_number += float(value[0])

    action_list = action_newcome + action_on

    total_cnt = np.array([float(obs[action][0]) for action in action_list])
    success_cnt = np.array([float(obs[action][1]) for action in action_list])
           
    def posterior(w): 
        
        # prior
        prior_log = 0.0

        if odds_ratios_only:
            if len(action_on) >= 2:
                w_mu_diff = np.array(w[-len(action_on):-1]) - par.mu[:-1]
                prior_log = \
                    -0.5 * w_mu_diff.dot(par.sigma_inv[:-1,:-1]).dot(np.transpose(w_mu_diff)) / total_number

        else:
            if len(action_on) >= 1:
                w_mu_diff = np.array(w[-len(action_on):]) - par.mu
                prior_log = \
                    -0.5 * w_mu_diff.dot(par.sigma_inv).dot(np.transpose(w_mu_diff)) / total_number

        # likelihood
        # 
        w_vec = np.array([w[i] + w[-1] for i in range(len(obs)-1)])
        w_vec = np.append(w_vec, w[-1])

        p = np.array([logistic(w[i] + w[-1]) for i in range(len(obs)-1)])
        p = np.append(p, logistic(w[-1]))
        p = np.maximum(.0001, p)
        p = np.minimum(.9999, p)

        likelihood_k = (success_cnt * np.log(p) + (total_cnt - success_cnt) * np.log(1. - p)) / total_number
        likelihood_log = np.sum(likelihood_k)

        posterior_log = (1. - discount) * prior_log + likelihood_log
        return posterior_log

    def gradient(w):
        p = np.array([logistic(w[i] + w[-1]) for i in range(len(obs) - 1)])
        p = np.append(p, logistic(w[-1]))

        # prior part
        gradient_prior = np.zeros(len(obs))

        if odds_ratios_only:
            if len(action_on) >= 2:
                gradient_prior[-len(action_on):-1] = \
                    - np.matmul(np.array(w[-len(action_on):-1]) - par.mu[:-1], par.sigma_inv[:-1,:-1]) / total_number

        else:
            if len(action_on) >= 1:
                gradient_prior[-len(action_on):] = - np.matmul(np.array(w[-len(action_on)]) - par.mu, par.sigma_inv) / total_number


        # likelihood part
        gradient_likelihood = (success_cnt * (1. - p) - (total_cnt - success_cnt) * p) / total_number
        gradient_likelihood[-1] = sum(gradient_likelihood)

        gradient = (1. - discount) * gradient_prior + gradient_likelihood
        return gradient

    return posterior, gradient



def extend(par1, par2, discount = 0.0):
    # align to same ref
    if len(par1.get_models()) == 0:
        return par2
    else:
        assert par1.get_models()[-1] == par2.get_models()[-1]

        action_not_observed = [x for x in par1.get_models() if x not in par2.get_models()]

        n_not_observed = len(action_not_observed)

        par1_sub = par1.transform2(action_not_observed + [par2.get_models()[-1]])
        action_list = par1_sub.get_models()[:-1] + par2.get_models() 

        mu = np.append(par1_sub.mu[:-1], par2.mu)
        sigma_inv = np.zeros((len(action_list), len(action_list)))

        sigma_inv[:n_not_observed, :n_not_observed] = (1. - discount) * par1_sub.sigma_inv[:-1, :-1]
        sigma_inv[:n_not_observed, -1] = (1. - discount) * par1_sub.sigma_inv[:-1, -1]
        sigma_inv[-1, :n_not_observed] = (1. - discount) * par1_sub.sigma_inv[-1, :-1]

        sigma_inv[n_not_observed:, n_not_observed:] = par2.sigma_inv
        
        out = LogisticBandit()
        out.reset(mu, sigma_inv, action_list)
    return out


