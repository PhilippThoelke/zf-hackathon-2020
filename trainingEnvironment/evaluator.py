from geneticAlgorithm import ANN, GeneticAlgorithm
from dataProcessing import ProfileManager
from environment import Simulator
import torch
import pandas as pd
from hyperparameters import *

print(f'Loading model "{MODEL_PATH.split("/")[-1]}"')
model = GeneticAlgorithm.load_model(MODEL_PATH)
print(f'Loading evaluation road profile "{ROAD_PROFILE_EVAL}"')
road_profile = ProfileManager.csv_to_profile(ROAD_PROFILE_EVAL, VELOCITY_eval)

history = []

env = Simulator(road_profile)
x = env.states[-1]
print('Simulating...')
for step in range(len(road_profile) - 1):
    Zb, Zb_dt, Zb_dtdt, Zt, Tz_dt, Zt_dtdt, last_i, Zh, Zh_dt = x
    x_torch = torch.from_numpy(x.reshape((1,) + x.shape))
    i = model(x_torch)[0,0].detach().numpy() * 2
    t = DT * step
    history.append([t, Zh, Zt, Zb, Zt_dtdt, Zb_dtdt, i])
    x = env.next(i)

df = pd.DataFrame(data=history, columns=['t', 'Zh', 'Zt', 'Zb', 'Zt_dtdt', 'Zb_dtdt', 'i'])
print(df)
df.to_csv('result.csv')
