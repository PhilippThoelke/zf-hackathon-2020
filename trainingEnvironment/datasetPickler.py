import pickle
from dataProcessing import ProfileManager

FILE = 'ts4_k_20.0.csv'
VELOCITY = 27

road_profile = ProfileManager.csv_to_profile(FILE, VELOCITY)[0]

with open(FILE + '.pickle', 'wb') as file:
    pickle.dump(road_profile)
