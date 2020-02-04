import numpy as np
import numpy.random 
import pandas as pd

from logisticbandit import LogisticBandit
from utils import logistic
from ts import TSPar


TIMESTEP = 50
def simulate_noise(p_list_seq, method, N=100):
    # p_list_seq: 
    
    if method in ["logistic_full", "logistic_or"]:
        fullpar = LogisticBandit()
    else:
        fullpar = TSPar()

    full_track = []
    obs_track = []


    for p_list in p_list_seq:

        prop = fullpar.win_prop(list(p_list.keys()))

        # generate data
        np_dict = {action: [np.round(N * np.array(prop[action])), p_list[action]] for action in p_list.keys()}
        obs_ts = simulate_obs(np_dict)

        obs_track.append(obs_ts)

        if method == "logistic_full": 
            fullpar.update(obs_ts, odds_ratios_only = False)
        elif method is not "logistic_full":
            fullpar.update(obs_ts)

        # regret
        max_p = max(p_list.values())
        regret = 0.
        for action, p in p_list.items():
            if p < max_p:
                regret += np_dict[action][0] * (max_p - p)

        full_track.append(regret)

    return full_track, obs_track


def simulate_constant(p_list, method, obs_list = [], N=100):
    
    if method in ["logistic_full", "logistic_or"]:
        fullpar = LogisticBandit()
    else:
        fullpar = TSPar()

    full_track = []
    obs_track = []

    p_full = {key: 1./len(p_list) for key in p_list.keys()}

    if len(obs_list) > 0:
        iterator = range(obs_list)
    else:
        iterator = range(TIMESTEP)

    for i in iterator:
        
        if len(obs_list) > 0:
            obs_ts = obs_list[i]
        else:
            # generate data
            np_dict = {action: [np.round(N * np.array(p_full[action])), p_list[action]] for action in p_list.keys()}
            obs_ts = simulate_obs(np_dict)

        obs_track.append(obs_ts)

        if method == "logistic_full": 
            fullpar.update(obs_ts, odds_ratios_only = False)
        elif method is not "logistic_full":
            fullpar.update(obs_ts)

        p_full = fullpar.win_prop()

        # regret
        max_p = max(p_list.values())
        regret = 0
        for action in p_list.keys():
            if p_list[action] < max_p:
                regret += obs_ts[action][0]

        full_track.append(regret)

    return full_track, obs_track


def simulate(p_list, MAX_TIMESTEP, noise = 0.0, N = 100):
    
    ortspar = LogisticBandit() 
    fullpar = LogisticBandit()
    tspar = TSPar()
    
    orts_track = []
    full_track = []
    ts_track = []

    n_ts = (float(N) * np.repeat(1.0 / 10.0, 10)).astype(int)

    p_noise = add_noise(p_list, noise = noise)

    model_ids = ["model_" + str(num) for num in range(10)]

    np_dict = {model: [n_ts[i], p_noise[i]] for i, model in enumerate(model_ids)} 
    obs_ts = simulate_obs(np_dict)
    obs_track = [obs_ts]

    # print(obs_ts)
    ortspar.update(obs_ts)
    fullpar.update(obs_ts, odds_ratios_only = False)
    tspar.update(obs_ts)

    for i in range(MAX_TIMESTEP):
        p_orts = ortspar.win_prop()
        p_full = fullpar.win_prop()
        p_ts   = tspar.win_prop()

        p_noise_ = add_noise(np.array(p_list), noise = noise)
        p_noise = {action: p_noise_[i] for i, action in enumerate(model_ids)}

        np_orts = {action: [np.round(N * np.array(p_orts[action])), p_noise[action]] \
            for action in ortspar.get_models()}

        np_full = {action: [np.round(N * np.array(p_full[action])), p_noise[action]]\
            for action in fullpar.get_models()}

        np_ts   = {action: [np.round(N * np.array(p_ts[action])), p_noise[action]]\
            for action in tspar.get_models()}

        # regret
        orts_track.append((1. - p_orts["model_9"]) * (p_noise["model_9"] - p_noise["model_0"]) * N)
        full_track.append((1. - p_full["model_9"]) * (p_noise["model_9"] - p_noise["model_0"]) * N)
        ts_track.append(  (1. - p_ts["model_9"]) * (p_noise["model_9"] - p_noise["model_0"]) * N)

        obs1 = simulate_obs(np_orts)
        obs2 = simulate_obs(np_full)
        obs3 = simulate_obs(np_ts)

        obs_track.append(obs2)
#        ts_track.append(N * max(p_noise.values()) - np.sum([obs1[action][1] for action in obs1.keys()]))
#        orts2_track.append(N * max(p_noise.values()) - np.sum([obs3[action][1] for action in obs3.keys()]))

        try:
            temp = ortspar.action_list, ortspar.mu, ortspar.sigma_inv
            ortspar.update(obs1)

        except:
            print("orts", temp, obs1)

        try:
            temp = fullpar.action_list, fullpar.mu, fullpar.sigma_inv
            fullpar.update(obs2, odds_ratios_only = False)

        except:
            print(obs_track)
            raise

        try:
            temp = tspar.action_list, tspar.alpha, tspar.beta
            tspar.update(obs3)

        except:
            print("tspar", temp, obs3)

    return orts_track, full_track, ts_track

# def generate_p(num_K = 10, epsilon = .1, noise = .0):
#     p_list = np.repeat(.5, num_K)
#     p_list[0] += epsilon
#     return add_noise(p_list, noise)


def simulate_one(p_list, MAX_TIMESTEP, noise = 0., N = 100):
    
    # np.random.seed(random_seed)

    fullpar = LogisticBandit()
    
    full_track = []

    n_ts = (float(N) * np.repeat(1.0 / 10.0, 10)).astype(int)

    p_noise = add_noise(p = p_list, noise = noise)

    model_ids = ["model_" + str(num) for num in range(10)]

    np_dict = {model: [n_ts[i], p_noise[i]] for i, model in enumerate(model_ids)} 
    obs_ts = simulate_obs(np_dict)
    obs_track = [obs_ts]


    fullpar.update(obs_ts, odds_ratios_only = False)

    for i in range(MAX_TIMESTEP):
        # p_orts = ortspar.win_prop()
        p_full = fullpar.win_prop()
        # p_ts   = tspar.win_prop()

        p_noise_ = add_noise(p = np.array(p_list), noise = noise)
        p_noise = {action: p_noise_[i] for i, action in enumerate(model_ids)}

        np_full = {action: [np.round(N * np.array(p_full[action])), p_noise[action]]\
            for action in fullpar.get_models()}

        # regret
        # orts_track.append((1. - p_orts["model_1"]) * (p_noise["model_1"] - p_noise["model_0"]) * N)
        full_track.append((1. - p_full["model_9"]) * (p_noise["model_9"] - p_noise["model_0"]) * N)
        # ts_track.append(  (1. - p_ts["model_1"]) * (p_noise["model_1"] - p_noise["model_0"]) * N)

        # obs1 = simulate_obs(np_orts)
        obs2 = simulate_obs(np_full)
        obs_track.append(obs2)

        # obs3 = simulate_obs(np_ts)

#        ts_track.append(N * max(p_noise.values()) - np.sum([obs1[action][1] for action in obs1.keys()]))
#        orts2_track.append(N * max(p_noise.values()) - np.sum([obs3[action][1] for action in obs3.keys()]))

        # temp = fullpar.action_list, fullpar.mu, fullpar.sigma_inv
        fullpar.update(obs2, odds_ratios_only = False)

    return full_track, obs_track



def add_noise(p_dict, noise, common=True):

    out = {}

    if common:
        random_w = noise * np.random.normal(0.0, 1.0)
        for key in p_dict.keys():
            p = p_dict[key]
            base_w = np.log(p / (1.0 - p))
            out[key] = logistic(base_w + random_w)
    else:
        for key in p_dict.keys():
            p = p_dict[key]
            base_w = np.log(p / (1.0 - p))
            random_w = noise * np.random.normal(0.0, 1.0)
            out[key] = logistic(base_w + random_w)

    return out


def simulate_obs(np_dict):
    out_dict = {}

    for action in np_dict.keys():
        n = np_dict[action][0]
        p = np_dict[action][1]

        out_dict[action] = [n, np.random.binomial(n, p)]
        
    return out_dict



def counterfactual(data, object, fullrank = False, decay = 0, aggressive = 1):
    # data is pandas dataframe: date, group, ctr, view_count

    regret_list = []
    ctr_total_list = []
    click_list = []
    arms_list = list(data.iloc[:,1].unique())
    prop_list = pd.DataFrame(columns = arms_list + ['round'])

    round_series = data.iloc[:,0].unique()
    for i, cur in enumerate(round_series):
    
        data_one = data.loc[data.iloc[:,0] == cur]
        N = sum(data_one.iloc[:,3])

        prop_dict = get_proportion(i, data_one, N, arms_list, object) # as value

        obs_sim = {}
        pv_sum = 0
        click_sum = 0
        max_ctr = 0.
        for k, v in prop_dict.items():
            ctr = data_one[data_one.iloc[:,1] == k].iloc[0,2]
            max_ctr = max(max_ctr, ctr)
            click = int(v * ctr)
            obs_sim[k] = v, click
            pv_sum += v
            click_sum += click

        prop_dict['round'] = cur
        prop_list = prop_list.append(prop_dict, ignore_index=True)

        regret = int(pv_sum * max_ctr) - click_sum
        regret_list.append(regret)
        ctr_total = click_sum / pv_sum
        ctr_total_list.append(ctr_total)
        click_list.append(click_sum)

        if fullrank:
            object.update(obs_sim, decay = decay, odds_ratios_only=False)
        else:
            object.update(obs_sim, decay = decay)

    return click_list, prop_list, regret_list, ctr_total_list


def get_proportion(i, data_one, N, arms_list, object):

    if i == 0:
        pv_dict = dict()
        for group_id in arms_list:
            props = 1./len(arms_list)
            pv_dict[group_id] = int(N*props)

    else:
        pv_dict = {k: int(v * N) for k, v in object.win_prop().items()}
 
    return pv_dict