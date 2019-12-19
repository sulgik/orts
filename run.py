#########
# unit test
########
from logisticbandit import LogisticBandit, is_pos_semidef
from simulate import simulate_constant, simulate, simulate_one, add_noise, simulate_obs
import numpy as np
from numpy.linalg import pinv, inv
from ts import TSPar
import numpy.random

############################
# using simulate function
############################
BASE_P = .5
p_list = {"arm_"+str(i): BASE_P for i in range(10)}
p_list["arm_9"] +=.01

N = 10000
n_rep = 200
NOISE = 0.

regret_fu = []
obs_rep = []
seed = []

for i in range(n_rep):
    try:
        k = np.random.randint(1, 1000)
        seed.append(k)
        np.random.seed(k)
        out = simulate_constant(
            p_list, "logistic_or", N = N)
    
    except:
        print("random seed:", k)

    print(i+1, "/", n_rep)

    regret_fu.append(out[0])
    obs_rep.append(out[1])

temp = np.asarray(regret_fu)
np.savetxt("simul/OR_constant.txt", temp)


temp.mean(axis = 1)



############


regret_or = []
regret_fu = []
regret_ts = []


for i in range(n_rep):
    try:
        k = np.random.randint(1, 10000)
        np.random.seed(k)

        out = simulate(
            p_list, MAX_TIMESTEP = TIMESTEP, noise = NOISE, N = N)
        print(i+1, "/", n_rep)
        regret_or.append(out[0])
        regret_fu.append(out[1])
        regret_ts.append(out[2])
    except:
        print("random seed:", k)
        break


regretOR = np.asarray(regret_or)
regretFU = np.asarray(regret_fu)
regretTS = np.asarray(regret_ts)

np.savetxt("simul/0_5_OR2.txt", regretOR)
np.savetxt("simul/0_5_FU2.txt", regretFU)
np.savetxt("simul/0_5_TS2.txt", regretTS)


regretOR.mean(axis = 0)
regretFU.mean(axis = 0)
regretTS.mean(axis = 0)

regretOR.median(axis = 0)
regretFU.median(axis = 0)
regretTS.median(axis = 0)


###############

regret_fu = []
obs_rep = []
seed = []

for i in range(n_rep):
    try:
        k = np.random.randint(1, 1000)
        seed.append(k)
        np.random.seed(k)
        out = simulate_one(
            p_list, MAX_TIMESTEP = TIMESTEP, noise = NOISE, N = N)
    
    except:
        print("random seed:", k)

    print(i+1, "/", n_rep)
    regret_fu.append(out[0])
    obs_rep.append(out[1])


temp = np.asarray(regret_fu)
temp.mean(axis = 1)



######
np.random.seed(732)
out = simulate_one(p_list, MAX_TIMESTEP = TIMESTEP, noise = NOISE, N = N)
out[0]
out[1]


########
obs_list = obs_rep[17]  # 30
fullpar = LogisticBandit()

t = -1

# 23
while t < 23:
    t += 1
    # i
    # obs_list[i]
    obs = out[1][t]
    print(obs)
    fullpar.update(obs, odds_ratios_only = False)
    fullpar.win_prop()

fullpar.mu
# fullpar.sigma_inv

obs = out[1][24]
obs
fullpar.update(obs, odds_ratios_only = False)
fullpar.mu

fullpar.win_prop()


##################
## two step 
##################
obs = {"model_0": [10, 3], "model_1": [10, 5], "model_2": [10, 2]}

fullpar = LogisticBandit()
orpar = LogisticBandit()

fullpar.update(obs, odds_ratios_only = False)
orpar.update(obs)

fullpar.win_prop()
orpar.win_prop()


obs_raw = {"model_0": [0,0], "model_1": [10, 5], "model_2": [10, 2]}
fullpar.update(obs_raw, odds_ratios_only = False)
orpar.update(obs_raw)

fullpar.win_prop()
orpar.win_prop()



orpar.update(obs_raw)

fullpar.win_prop()
orpar.win_prop()



## issue with odd_ratios_only = False
N = 2000

ortspar2 = LogisticBandit()

n_ts = (float(N) * np.repeat(1.0 / 10.0, 10)).astype(int)

p_list = np.append([.5, .51, .5, .5, .51], [.5] * 5)
noise = 0.0

out = simulate(p_list, MAX_TIMESTEP = 5, noise = 0., N = 2000)

p_noise = add_noise(p_list, noise = noise)

model_ids = ["model_" + str(num) for num in range(10)]

np_dict = {model: [n_ts[i], p_noise[i]] for i, model in enumerate(model_ids)} 
obs_ts = simulate_obs(np_dict)

tspar.update(obs_ts) 
ortspar2.update(obs_ts, odds_ratios_only = False)

for i in range(MAX_TIMESTEP):
    p_ts = tspar.win_prop()
    p_orts2 = ortspar2.win_prop()

    p_noise_ = add_noise(np.array(p_list), noise = noise)
    p_noise = {action: p_noise_[i] for i, action in enumerate(model_ids)}

    np_ts = {action: [np.round(N * np.array(p_ts[action])), p_noise[action]] \
        for action in tspar.get_models()}

    np_orts2 = {action: [np.round(N * np.array(p_orts2[action])), p_noise[action]]\
        for action in ortspar2.get_models()}

    obs1 = simulate_obs(np_ts)
    obs3 = simulate_obs(np_orts2)

#        ts_track.append(N * max(p_noise.values()) - np.sum([obs1[action][1] for action in obs1.keys()]))
#        orts2_track.append(N * max(p_noise.values()) - np.sum([obs3[action][1] for action in obs3.keys()]))

    ts_track.append(   1. - p_ts["model_1"]    - p_ts["model_4"])
    orts2_track.append(1. - p_orts2["model_1"] - p_orts2["model_4"])

    tspar.update(obs1)
    ortspar2.update(obs3, odds_ratios_only = False)
    





##
par = fullpar.get_transformed(["model_1", "model_2"])

obs = {i:obs_raw[i] for i in obs_raw.keys() if obs_raw[i][0] > 0}

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

actions_list = action_newcome + action_on

# mu
if len(action_on) == 0:
    par_sub = LogisticBandit() 
else:    
    par_sub = par
    par_sub.transform(action_on)

target_fn, gradient_fn = _build_fns(par_sub, obs, action_newcome, action_on, odds_ratios_only= False, discount=0.)

if len(action_on) >= 1:
    initial = np.concatenate((\
        np.repeat(0., len(actions_list) - len(par_sub.mu)),\
        par_sub.mu[:-1],\
        [np.log(p_ref / (1. - p_ref))]))

target_fn_or, gradient_fn_or = _build_fns(par_sub, obs, action_newcome, action_on, odds_ratios_only = True, discount=0.)

p_ref = float(obs[actions_list[-1]][1]) / float(obs[actions_list[-1]][0])
print(p_ref)

initial = np.concatenate((\
    np.repeat(0., len(actions_list) - len(par_sub.mu)),\
    par_sub.mu))

print("initial:", initial)

mu_hat_full = \
    estimate_mu(target_fn, gradient_fn,\
        initial = initial, alpha_0 =.01, max_iter = 50000)

mu_hat_or = \
    estimate_mu(target_fn_or, gradient_fn_or,\
        initial = initial, alpha_0 = .01, max_iter = 50000)



mu_hat_full
mu_hat_or

fullpar.update(obs_raw, odds_ratios_only = False)
orpar.update(obs_raw)

fullpar.mu
orpar.mu




# orpar.update(obs_orts)




###################

# simulation

import multiprocessing
from datetime import datetime

n_rep = 2
TIMESTEP = 13

def f(noise):
    print("1 loop")
    return simulate(p_list, MAX_TIMESTEP = TIMESTEP, noise = noise, N = N)

pool = multiprocessing.Pool(processes = 2)

rate_list = np.repeat([0.0], n_rep).tolist()

start_time = datetime.now()
result_list = []
result_list = pool.map(f, rate_list)
datetime.now() - start_time
# datetime.timedelta(seconds=14756, microseconds=364976)


# visualize
regret_or = np.empty((TIMESTEP))
regret_full = np.empty((TIMESTEP))

for i in range(n_rep):
    regret_or = np.vstack((regret_or, np.array(result_list[2*i+1][0])))
    regret_full = np.vstack((regret_full, np.array(result_list[2*i+1][1])))

np.mean(regret_or, axis=0)
np.mean(regret_full, axis=0)

np.median(regret_or, axis=0)
np.median(regret_full, axis=0)


regret_or = np.empty((TIMESTEP))
regret_full = np.empty((TIMESTEP))










# positive semi definite

sigma_inv = np.array([[ 810.13459259,    0.         ,   0.,            0.      ,      0.,
     0.        ,    0.          ,  0.  ,          0.      ,    810.13459259],
 [   0.        ,  243.13108878   , 0.   ,         0.       ,     0.,
     0.         ,   0.,            0.    ,        0.       ,   243.13108878],
 [   0.          ,  0. ,         184.70741759,    0.        ,    0.,
     0.           , 0.  ,          0.  ,          0.        ,  184.70741759],
 [   0.           , 0.   ,         0.   ,       391.74994476 ,   0.,
     0.,            0.    ,        0.    ,        0.,          391.74994476],
 [   0. ,           0.     ,       0.     ,       0. ,        1399.76399889,
     0.  ,          0.      ,      0.      ,      0.  ,       1399.76399889],
 [   0.   ,         0.       ,     0.       ,     0.   ,         0.,
   259.83627693,    0.        ,    0.        ,    0.    ,      259.83627693],
 [   0.         ,   0.         ,   0.         ,   0.     ,       0.,
     0.         ,1039.04745322  ,  0.          ,  0.      ,   1039.04745322],
 [   0.          ,  0.           , 0.           , 0.       ,     0.,
     0.           , 0. ,         213.10164436,    0.        ,  213.10164436],
 [   0.           , 0.  ,          0.    ,        0.         ,   0.,
     0.           , 0.   ,         0.     ,     221.95391378 , 221.95391378],
 [ 810.13459259 , 243.13108878 , 184.70741759,  391.74994476 ,1399.76399889,
   259.83627693 ,1039.04745322 , 213.10164436 , 221.95391378 , 446.862     ]])



# visualize
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt


data = pd.DataFrame({"t":range(len(out[0])), "odds_ratios": out[0], "full_rank": out[1]})

plt.plot(out[0])
plt.show()




## reproduce singular matrix and fix
lb_full = LogisticBandit()
lb = LogisticBandit()
obs = {'model_0': [100.0, 50], 'model_1': [100.0, 30.], 'model_2': [100., 60.]}
#npq1 = 100 * .5* .5 = 25, npq2 = , npq3 = 21, npq3 = 24
lb.update(obs)
lb_full.update(obs, odds_ratios_only = False)

lb_full.mu
lb_full.sigma_inv
get_par_(lb_full, ['model_2', 'model_0', 'model_1'])







np.transpose(transform_mat2)
transform_mat2

# use sigma
a = np.matmul(transform_mat2, inv(lb_full.sigma_inv), np.transpose(transform_mat2))
a
inv(a)


lb_full.sigma_inv

transform_mat2

lb_full.sigma_inv

# use sigma_inv
np.matmul(inv(np.transpose(transform_mat)), lb_full.sigma_inv, inv(transform_mat))



lb_full.get_par(['model_2', 'model_0', 'model_1'])[1]

lb_full.sigma_inv


obs = {'model_0': [10.0, 5.], 'model_1': [10., 3.], 'model_2': [0., 0.]}
# + 2.5, + 2.1, + 24
lb.update(obs)
lb_full.update(obs, odds_ratios_only = False)

lb_full.get_models()
lb_full.sigma_inv


lb_full.win_prop()



lb.sigma_inv









# obs1 = {'model_0': [14.0, 6], 'model_1': [164.0, 82], 'model_3': [1.0, 0], 'model_4': [2.0, 2], 'model_5': [12.0, 5], 'model_6': [1.0, 1], 'model_8': [2.0, 1], 'model_9': [3.0, 0]}
lb = LogisticBandit()
obs = {'model_0': [14.0, 6], 'model_1': [164.0, 82.], 'model_2': [4.0, 2.]}
lb.update(obs, odds_ratios_only = False)
lb.win_prop()

# sigma inv
lb.sigma_inv
# sigma
np.linalg.pinv(lb.sigma_inv)

obs = {'model_1':[100.,20.], 'model_2':[10.,2.]}

lb.update(obs, odds_ratios_only = False)
lb.get_models()





lb = LogisticBandit()
lb.update({"a":[10.,3.], "b": [20.,2.], "c":[10., 4.]})
lb.sigma_inv
np.linalg.pinv(lb.sigma_inv)
is_pos_semidef(np.linalg.pinv(lb.sigma_inv))

lb.update({'a':[0.,0.], "b": [10., 2.], 'c':[2.,1.], 'd':[10.,5.]})
lb.action_list

np.linalg.pinv(lb.sigma_inv)
is_pos_semidef(np.linalg.pinv(lb.sigma_inv))
out








par = LogisticBandit()
obs = {"m1": [100, 20], "m2": [100, 18]}

estimate(par = par, obs= obs, alpha_0=.3, max_iter = 2, odds_ratios_only = True)






#####
# simulation 
#####


############
# TO DO: overflow, underflow in np.exp
######

MAX_TIMESTEP  = 100


def f(noise):
    print("start:", datetime.datetime.now())
    out = simulate(MAX_TIMESTEP, noise, N = 100)
    print("end:", datetime.datetime.now())
    return out

pool = multiprocessing.Pool(processes = 2)

noise_list = [0.3]
start_time = datetime.datetime.now()
result_list = pool.map(f, noise_list * 100)
datetime.datetime.now() - start_time














regret = [np.array(result_list[0][0]),
np.array(result_list[0][1])]


for result in result_list[1:]:
    regret[0] = np.vstack((regret[0], np.array(result[0])))
    regret[1] = np.vstack((regret[1], np.array(result[1])))


#    regret[2] = np.vstack((regret[2], np.array(result[2])))

regret[0].mean(axis = 0).sum()
regret[1].mean(axis = 0).sum()






regret[2].mean(axis = 0).sum()

















ortspar2 = LogisticBandit()
ortspar2.update({"model_1": [100, 20], "model_2": [100, 30]})
ortspar2.mu
ortspar2.sigma_inv
ortspar2.update({"model_1": [100, 18]})
ortspar2.mu
ortspar2.sigma_inv




# np.seterr(all='raise')
simulate(MAX_TIMESTEP = 1000, noise = .3)














ortspar2 = LogisticBandit()

# iteration
N = 100
noise = 0.3

n_ts = float(N) * np.repeat(1.0 / 10.0, 10)
p_list = np.array([.51] + [.5] * 9)

p_noise = add_noise(np.array(p_list), noise = noise)

model_ids = ["model_" + str(num) for num in range(10)]

np_dict = {model: [n_ts[i], p_noise[i]] for i, model in enumerate(model_ids)} 
obs_ts = simulate_obs(np_dict)

ortspar2.update(obs_ts, odds_ratios_only = False)


p_orts2 = ortspar2.win_prop(aggressive = 1.0)

p_noise_ = add_noise(np.array(p_list), noise = noise)
p_noise = {action: p_noise_[i] for i, action in enumerate(model_ids)}

np_orts2 = {action: [N * np.array(p_orts2[action]), p_noise[action]] for action in ortspar2.get_models()}

obs3 = simulate_obs(np_orts2)
    
ortspar2.update(obs3, odds_ratios_only = False)
















from ts import LogisticBandit, logistic
import numpy as np

p_list = [.3, .33, .32, .3]











def _obs_regret(obs_oracle, n_ts):
    reward_oracle = max(obs_oracle)

    out = []
    for i in len(n_ts):    
        out.append(np.random.binomial(obs_oracle[0], float(n_ts[0]) / float(sum(n_ts))))
    
    obs = {"model_1": [sum(n_ts), n_ts[0]], "model_2": [sum(n_ts), n_ts[1]]}
    
    regret = reward_oracle - out1 - out2

    return zip(obs, regret)





import numpy as np
from numpy.linalg import inv
    
import importlib
from datetime import datetime

import pickle
import pandas as pd













import cProfile

#####################
# regret analysis 
#####################






p_list

mc = np.random.binomial(total_size)


out = []
for i in len(n_ts):    
    out.append(np.random.binomial(obs_oracle[0], float(n_ts[0]) / float(sum(n_ts))))

obs = {"model_1": [sum(n_ts), n_ts[0]], "model_2": [sum(n_ts), n_ts[1]]}

regret = reward_oracle - out1 - out2

return zip(obs, regret)


def obs_regret(sizes_list, p_list):
    n = len(n_ts)


    out = []
    for i in len(n_ts):    
        out.append(np.random.binomial(obs_oracle[0], float(n_ts[0]) / float(sum(n_ts))))
    
    obs = {"model_1": [sum(n_ts), n_ts[0]], "model_2": [sum(n_ts), n_ts[1]]}
    
    regret = reward_oracle - out1 - out2

    return zip(obs, regret)



#####################
# regret analysis 
#####################

# simulation
action_list = ["model_1", "model_2", "model_3"]

rate = 0.0

N = 2000


p_ground_truth = {'model_1': .11, "model_2": .1, "model_3": .09}
p_list = [i for i in p_ground_truth.values()]

p_noise = add_noise(np.array(p_list), rate = rate)

n_ts = np.repeat(int(N / float(len(p_list))), len(p_list))

obs_raw = simulate_obs(zip(n_ts, p_noise))
obs = {}
for i, key in enumerate(p_ground_truth.keys()):
    obs[key] = obs_raw[i]



ortspar2 = ORPar2()
ortspar2.first(obs)

win_prop2(ortspar2)



ortspar.first(obs)
win_prop(ortspar)





ortspar2.Sigma




simulate_regret(3, 0.0)


ortspar.first(obs)

ortspar.mu
ortspar.Sigma

p_orts = win_prop(ortspar)
p_orts




# simulation
TIME_STEP = 7
p_list = [i for i in p_ground_truth.values()]
NOISE = 0.0



import multiprocessing

def f(rate):
    print("1 loop")
    return simulate_regret(TIME_STEP, rate)

pool = multiprocessing.Pool(processes = 2)

# rate_list = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]
rate_list = [0.0]
start_time = datetime.now()
result_list = pool.map(f, rate_list * 10)
datetime.now() - start_time
# 5.5 hours

result_list[0]


# save
f = open("regret_zero2.pkl", "wb")
pickle.dump(result_list, f)
f.close()




# sumamry
df1_list = [pd.DataFrame({})] * 1
df2_list = [pd.DataFrame({})] * 1

i = 0
for onesim in result_list:
    j = 0
    df1_list[j] = df1_list[j].append(pd.DataFrame([onesim[0]]))
    df2_list[j] = df2_list[j].append(pd.DataFrame([onesim[1]]))
    i += 1

df1_list[0].mean(0)
df2_list[0].mean(0)


output = pd.DataFrame(
{'rate': rate_list,
'logistic': [df1_list[i].mean(0).sum(0) for i in range(6)],
'robust': [df2_list[i].mean(0).sum(0) for i in range(6)]})








# load
file = open('output.pkl', 'rb')
result_list_loaded = pickle.load(file)
file.close()

result_list_loaded['0.0'][0]









# visualize
import seaborn as sns
import matplotlib.pyplot as plt 


x = np.linspace(0, 20, 100)  # Create a list of evenly-spaced numbers over the range
plt.plot(x, np.sin(x))       # Plot the sine of each x point
plt.show()  










# overlap just one
par = OddsRatioParameters()
obs = {"model_1": [100, 50], 'model_2': [100, 51]}
par.first(obs)
win_prop(par)

obs = {"model_1":[1000, 500], "model_2":[1000, 497], "model_3":[1000, 480]}
par.update(obs)
par.Sigma
win_prop(par)

obs = {"model_1":[1000, 499], "model_2":[1000, 500]}
par.update(obs)
par.action_nonref
par.action_ref
par.mu
par.Sigma
win_prop(par)

# To Do: change ts using regular logistic bandit
import ts
from ts import *

importlib.reload(ts)

pp = TSPar()

obs = {"model_1": [100, 50], 'model_2': [100, 51]}
pp.first(obs)

pp.action_list 
pp.alpha
pp.beta

win_prop_ts(pp)


obs = {"model_1": [100, 50], 'model_2': [100, 51]}
pp.update(obs)
win_prop_ts(pp)















# REPORT: metric to compare: cumulative regret and final decision
cum_regret_ts = []
cum_regret_orts = []
final_ts = []
final_orts = []

for onesim in simulated_data:
    cum_regret_ts.append(sum(onesim[0]))
    cum_regret_orts.append(sum(onesim[1]))
    final_ts.append(onesim[0][-1])
    final_orts.append(onesim[1][-1])

cum_regret_ts
cum_regret_orts
sum(cum_regret_ts)
sum(cum_regret_orts)

final_ts
final_orts
sum(final_ts)
sum(final_orts)






# dynamic case








def simulate_one(MAX_TIMESTEP, rate):
    n_ts = np.repeat(N / 2.0, 2)
    n_orts = n_ts
    tspar = ORPar2()
    ortspar = OddsRatioParameters()
    p_noise = add_noise(np.array(p_list), rate = rate)
    obs_ts = simulate_obs(zip(n_ts, p_noise))
    tspar.first(obs_ts)
    ortspar.first(obs_ts)
    p_ts = win_prop2(tspar)
    p_orts = win_prop(ortspar)
    n_ts = N * np.array(p_ts[0])
    n_orts = N * np.array(p_orts[0])
    ts_track = [p_ts[0][0]]
    orts_track = [p_orts[0][0]]
    i = 0
    while(i < MAX_TIMESTEP):
        p_noise = add_noise(np.array(p_list), rate = rate)
        if int(n_ts[0]) == 0 or int(n_ts[0]) == N:
            ts_track.append(ts_track[-1])
        else:
            obs_ts = simulate_obs(zip(n_ts, p_noise))
            tspar.update(obs_ts)        
            p_ts = win_prop2(tspar)
            n_ts = N * np.array(p_ts[0])
            ts_track.append(p_ts[0][0])

        if int(n_orts[0]) == 0 or int(n_orts[0]) == N:
            orts_track.append(orts_track[-1])
        else:
            obs_orts = simulate_obs(zip(n_orts, p_noise))
            ortspar.update(obs_orts)
            p_orts = win_prop(ortspar)
            n_orts = N * np.array(p_orts[0])
            orts_track.append(p_orts[0][0])
        
        i += 1

    return ts_track, orts_track




def simulate_ts(MAX_TIMESTEP, rate):
    n_ts = np.repeat(N / 2.0, 2)
    tspar = TSPar()
    p_noise = add_noise(np.array(p_list), rate = rate)
    obs_ts = simulate_obs(zip(n_ts, p_noise))
    tspar.first(obs_ts)
    p_ts = win_prop_ts(tspar)
    n_ts = N * np.array(p_ts[0])
    ts_track = [p_ts[0][0]]
    i = 0
    while(i < MAX_TIMESTEP):
        p_noise = add_noise(np.array(p_list), rate = rate)
        if int(n_ts[0]) == 0 or int(n_ts[0]) == N:
            ts_track.append(ts_track[-1])
        else:
            obs_ts = simulate_obs(zip(n_ts, p_noise))
            tspar.update(obs_ts)        
            p_ts = win_prop_ts(tspar)
            n_ts = N * np.array(p_ts[0])
            ts_track.append(p_ts[0][0])
        
        i += 1
    return ts_track


