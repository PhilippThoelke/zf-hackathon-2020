import pickle
from dataProcessing import ProfileManager
from joblib import Parallel, delayed

FILE = 'ts4_k_20.0.csv'
VELOCITIES = [8.3, 13.8, 19.4, 27.7]

def run(vel):
    road_profile = ProfileManager.csv_to_profile(FILE, [vel])[0]
    with open('pickleData/' + FILE + f'_vel{vel}.pickle', 'wb') as file:
        pickle.dump(road_profile, file)

Parallel(n_jobs=2)(delayed(run)(vel) for vel in VELOCITIES)
