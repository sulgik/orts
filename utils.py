import numpy as np
from numpy import clip
from scipy.optimize import minimize


def estimate(prior, obs, indexes, discount = 0.):
    #
    # estimate mu and using this estimate sigma inverse
    #

    # mu
    target_fn, gradient_fn = _build_fns(prior, obs, indexes, discount)
    initial_mu = _compute_initial(prior, obs, indexes)

    ww = minimize(target_fn, initial_mu, jac = gradient_fn, method = 'Newton-CG')
    mu_hat = ww.x

    # sigma_inv
    sigma_inv = \
        _estimate_sigma_inv(mu_hat, prior, obs, indexes, discount)
        
    return mu_hat, sigma_inv


def _compute_initial(prior, obs, indexes):
    index_obs = \
        [i for i in range(indexes[0])] + [i for i in range(indexes[1], indexes[2])]

    initial = np.zeros(indexes[2])

    p_ref = np.clip(float(obs[-1][1]) / float(obs[-1][0]), .001, .999)
    # initial[-1] = np.log(p_ref / (1. - p_ref))

    for i, j in enumerate(index_obs[:-1]):
        p = np.clip(float(obs[i][1]) / float(obs[i][0]), .001, .999)
        initial[j] = np.log(p / (1. - p)) - np.log(p_ref / (1. - p_ref))
    
    if prior[0] is not None:
        # if len(prior[0]) > 1:
        initial[indexes[0]:] = prior[0]
            # initial[indexes[0]:] = initial[indexes[0]:] * .5 + prior[0] * .5

    return initial


def logistic(x):
    # set a truncation exponent.
    trunc = 9.  # exp(8)/(1+exp(8)) = 0.9997 which is close enough to 1 as to not matter in most cases.
    x = np.clip(x, -trunc, trunc)

    if x > 0:
        return 1. / (1. + np.exp(-x))
    else:
        return np.exp(x) / (1. + np.exp(x))


def is_pos_semidef(x):
    return np.all(np.linalg.eigvals(x) >= 0)


def _estimate_sigma_inv(mu_hat, prior, obs, indexes, discount):

    # sigma_inv prior part
    sigma_inv_prior = np.zeros((indexes[2], indexes[2]))
    sigma_inv_prior[indexes[0]:, indexes[0]:] = prior[1]

    # sigma_inv likelihood part
    index_obs = [i for i in range(indexes[0])] + [i for i in range(indexes[1], indexes[2])]

    p = np.array([logistic(mu + mu_hat[-1]) for mu in mu_hat[index_obs][:-1]])
    p_ref = logistic(mu_hat[-1])

    total_cnt = np.array([float(i[0]) for i in obs[:-1]])
    total_cnt_ref = float(obs[-1][0])

    npq = total_cnt * p * (1. - p)
    npq_ref = total_cnt_ref * p_ref * (1. - p_ref)
    sigma_inv_likelihood_full = np.zeros((indexes[2], indexes[2]))

    for i, j in enumerate(index_obs[:-1]):
        sigma_inv_likelihood_full[ j,  j] = npq[i]
        sigma_inv_likelihood_full[ j, -1] = npq[i]
        sigma_inv_likelihood_full[-1,  j] = npq[i]
    
    sigma_inv_likelihood_full[-1, -1] = np.sum(npq) + npq_ref

    sigma_inv = (1. - discount) * sigma_inv_prior + sigma_inv_likelihood_full
        
    return sigma_inv


def _build_fns(prior, obs, indexes, discount):
    #
    # target_fn, gradient_fn constructor
    #
    total_number = 0.0
    for i in obs:
        total_number += float(i[0])

    total_cnt   = np.array([float(i[0]) for i in obs])
    success_cnt = np.array([float(i[1]) for i in obs])

    index_prior = [i for i in range(indexes[0], indexes[2])]
    index_obs   = [i for i in range(indexes[0])] + [i for i in range(indexes[1], indexes[2])]

    def posterior(w): 
        # "negative" log likelihood which need to be maximize

        # prior
        prior_log = 0.

        if len(index_prior) > 0:
            w_mu_diff = np.array(w[index_prior]) - np.array(prior[0])
            prior_log = \
                0.5 * w_mu_diff.dot(prior[1]).dot(np.transpose(w_mu_diff)) / total_number

        # likelihood
        p = np.array([logistic(w[i] + w[-1]) for i in index_obs[:-1]])
        p = np.append(p, logistic(w[-1]))
        p = np.clip(p, .001, .999)

        likelihood_k = - (success_cnt * np.log(p) + (total_cnt - success_cnt) * np.log(1. - p)) / total_number
        likelihood_log = np.sum(likelihood_k)

        posterior_log = (1. - discount) * prior_log + likelihood_log

        return posterior_log

    def gradient(w):
        p = np.array([logistic(w[i] + w[-1]) for i in index_obs[:-1]])
        p = np.append(p, logistic(w[-1]))

        num_dim = indexes[2]

        # prior part
        grad_prior = np.zeros(num_dim)
        if len(index_prior) > 0:
            grad_prior[index_prior] =\
                np.matmul(np.array(w[index_prior]) - prior[0], prior[1]) / total_number

        # likelihood part
        grad_likelihood = np.zeros(num_dim)
        grad_likelihood[index_obs] = - (success_cnt * (1. - p) - (total_cnt - success_cnt) * p) / total_number
        grad_likelihood[-1] = np.sum(grad_likelihood)

        gradient_val = (1. - discount) * grad_prior + grad_likelihood
        return gradient_val
        
    return posterior, gradient
