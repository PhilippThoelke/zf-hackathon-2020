from geneticAlgorithm import ANN, GeneticAlgorithm
from dataProcessing import ProfileManager
from environment import Simulator
import torch
import pandas as pd
import numpy as np
from hyperparameters import *
from scipy import signal
import pickle

dt = 0.005

#road specific weight factor
tmp = ROAD_PROFILE_EVAL.split('_k_')[1].split('.csv')[0]
k = float(tmp)

Mb = 500        #mass quarter body [kg]
Mt = 50        	#mass tire + suspention system [kg]

def t_target(history):
    # extract Zb acceleration from the last N states
    Zb_dtdt = np.array(history)[:,5]

    #compute bandpass 2nd order from 0.4 - 3 Hz
    b, a = Simulator._butter_bandpass(0.4, 3, int(1 / dt), 2)
    zi = signal.lfilter_zi(b, a)
    z, _ = signal.lfilter(b, a, Zb_dtdt, zi=zi)

    #calculate variance alpha_1
    varZb_dtdt = np.var(z)

    #compute bandpass 2nd order from 0.4 - 3 Hz
    b, a = Simulator._butter_bandpass(10, 30, int(1 / dt), 1)
    zi = signal.lfilter_zi(b, a)
    z1, _ = signal.lfilter(b, a, Zb_dtdt, zi=zi)

    #calculate variance alpha_2
    varZb_dtdt_h = np.var(z1)

    #compute T_target
    target = k * varZb_dtdt_h + varZb_dtdt
    return target

def constraint_satisfied(history):
    # extract Zt acceleration from the last N states
    Zt_dtdt = np.array(history)[:,4]

    #standard deviation of Zt_dtdt
    devZt_dtdt = np.std(Zt_dtdt * Mt)

    #boundary condition
    F_stat_bound = (Mb + Mt) * 9.81 / 3.0
    return devZt_dtdt <= F_stat_bound

if __name__ == '__main__':
    print(f'Loading model "{MODEL_PATH.split("/")[-1]}"')
    model = GeneticAlgorithm.load_model(MODEL_PATH)
    print(f'Loading evaluation road profile "{ROAD_PROFILE_EVAL}"')
    with open(ROAD_PROFILE_EVAL, 'rb') as file:
        road_profile = pickle.load(file)

    history = []

    env = Simulator(road_profile)
    x = env.states[-1]
    print('Simulating...')
    for step in range(len(road_profile) - 1):
        Zb, Zb_dt, Zb_dtdt, Zt, Tz_dt, Zt_dtdt, last_i, Zh, Zh_dt = x

        # add a moving average of the last states to the model's input
        moving_avg = env.moving_average()
        x = np.concatenate((x, moving_avg))

        x_torch = torch.from_numpy(x.reshape((1,) + x.shape))
        i = model(x_torch)[0,0].detach().numpy() * 2

        t = DT * step
        history.append([t, Zh, Zt, Zb, Zt_dtdt, Zb_dtdt, i])

        # next step in the simulation
        x = env.next(i)

    print('T_target:', t_target(history))
    print('Constraint satisfied:', constraint_satisfied(history))

    df = pd.DataFrame(data=history, columns=['t', 'Zh', 'Zt', 'Zb', 'Zt_dtdt', 'Zb_dtdt', 'i'])
    print(df)
    df.to_csv('result.csv', index=False)
