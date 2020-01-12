import pickle
from dataProcessing import ProfileManager

FILE = 'ts4_k_20.0.csv'
VELOCITIES = [8, 20, 27]

for vel in VELOCITIES:
    road_profile = ProfileManager.csv_to_profile(FILE, [vel])[0]
    with open(FILE + f'_VEL_{vel}.pickle', 'wb') as file:
        pickle.dump(road_profile)
