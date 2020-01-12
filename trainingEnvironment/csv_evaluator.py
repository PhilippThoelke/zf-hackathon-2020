from geneticAlgorithm import ANN, GeneticAlgorithm
from dataProcessing import ProfileManager
from environment import Simulator
import torch
import pandas as pd
from hyperparameters import *

MODEL_DIRECTORY = '../models/'
MODEL_NAMES = ['grid_29.roadie', 'model_20.roadie', 'model_84.roadie']
ROAD_PROFILE_LIST = ['ts3_1_k_3.0.csv', 'ts3_2_k_3.0.csv', 'ts3_3_k_3.0.csv']

df = pd.DataFrame(data=history, columns=['t', 'Zh', 'Zt', 'Zb', 'Zt_dtdt', 'Zb_dtdt', 'i'])
print(df)
df.to_csv('result.csv')

for index in range(len(MODEL_NAMES)):
    for roads in range(len(ROAD_PROFILE_LIST)):

        print(f'Loading model "{MODEL_NAMES[index].split(".")[0]}"')
        model = GeneticAlgorithm.load_model(MODEL_DIRECTORY + MODEL_NAMES[index])
        print(f'Loading evaluation road profile "{ROAD_PROFILE_LIST[roads]}"')
        road_profile = ProfileManager.csv_to_profile(ROAD_PROFILE_LIST[roads], VELOCITY)

        history = []

        env = Simulator(road_profile)
        x = env.states[-1]
        print('Simulating...')
        for step in range(len(road_profile) - 1):
            Zb, Zb_dt, Zb_dtdt, Zt, Tz_dt, Zt_dtdt, last_i, Zh, Zh_dt = x
            x_torch = torch.from_numpy(x.reshape((1,) + x.shape))
            i = model(x_torch)[0,0].detach().numpy() * 2
            #i = index
            t = DT * step
            history.append([t, Zh, Zt, Zb, Zt_dtdt, Zb_dtdt, i])
            x = env.next(i)

        df = pd.DataFrame(data=history, columns=['t', 'Zh', 'Zt', 'Zb', 'Zt_dtdt', 'Zb_dtdt', 'i'])
        #print(df)
        df.to_csv(f'../results/{MODEL_NAMES[index].split(".")[0] + "_" + ROAD_PROFILE_LIST[roads].split(".")[0]}.csv')
