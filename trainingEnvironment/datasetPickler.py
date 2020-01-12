import pickle
from dataProcessing import ProfileManager

PATH = '../datasets/ts4_k_20.csv'
VELOCITY = 27

road_profile = ProfileManager.csv_to_profile(PATH, VELOCITY)[0]

with open(PATH.split('/')[-1] + '.pickle', 'wb') as file:
    pickle.dump(road_profile)
