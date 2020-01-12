from geneticAlgorithm import ANN, GeneticAlgorithm
from dataProcessing import ProfileManager
from environment import Simulator
import torch
import pandas as pd
import numpy as np
from scipy import signal

MODEL_PATH = '../models/2020_01_11-23_41_53-conv/model_37.roadie'
ROAD_PROFILE_FILE = 'ts3_1_k_3.0.csv'
VELOCITY = [27]
DT = 0.005
K = 3

Mb = 500        #mass quarter body [kg]
Mt = 50        	#mass tire + suspention system [kg]

def t_target(history):
    # extract Zb acceleration from the last N states
    Zb_dtdt = np.array(history)[:,5]

    #compute bandpass 2nd order from 0.4 - 3 Hz
    b, a = Simulator._butter_bandpass(0.4, 3, int(1 / DT), 2)
    zi = signal.lfilter_zi(b, a)
    z, _ = signal.lfilter(b, a, Zb_dtdt, zi=zi)

    #calculate variance alpha_1
    varZb_dtdt = np.var(z)

    #compute bandpass 2nd order from 0.4 - 3 Hz
    b, a = Simulator._butter_bandpass(10, 30, int(1 / DT), 1)
    zi = signal.lfilter_zi(b, a)
    z1, _ = signal.lfilter(b, a, Zb_dtdt, zi=zi)

    #calculate variance alpha_2
    varZb_dtdt_h = np.var(z1)

    #compute T_target
    target = K * varZb_dtdt_h + varZb_dtdt
    return target

def constraint_satisfied(history):
    # extract Zt acceleration from the last N states
    Zt_dtdt = np.array(history)[:,4]

    #standard deviation of Zt_dtdt
    devZt_dtdt = np.std(Zt_dtdt * Mt)

    #boundary condition
    F_stat_bound = (Mb + Mt) * 9.81 / 3.0
    return devZt_dtdt <= F_stat_bound

print(f'Loading model "{MODEL_PATH.split("/")[-1]}"')
model = GeneticAlgorithm.load_model(MODEL_PATH)
print(f'Loading evaluation road profile "{ROAD_PROFILE_FILE}"')
road_profile = ProfileManager.csv_to_profile(ROAD_PROFILE_FILE, VELOCITY)[0]

history = []

env = Simulator(road_profile)
x = env.states[-1]
print('Simulating...')
for step in range(len(road_profile) - 1):
    Zb, Zb_dt, Zb_dtdt, Zt, Tz_dt, Zt_dtdt, last_i, Zh, Zh_dt = x

    '''
    x = env.last_steps()
    x = x.reshape((1,) + x.shape)
    x = torch.from_numpy(x).permute(0, 2, 1)
    i = model(x)[0,0].detach().numpy() * 2
    '''
    i = 0.5

    t = DT * step
    history.append([t, Zh, Zt, Zb, Zt_dtdt, Zb_dtdt, i])
    x = env.next(i)

print('T_target:', t_target(history))
print('Constraint satisfied:', constraint_satisfied(history))

df = pd.DataFrame(data=history, columns=['t', 'Zh', 'Zt', 'Zb', 'Zt_dtdt', 'Zb_dtdt', 'i'])
print(df)
df.to_csv('result.csv')
