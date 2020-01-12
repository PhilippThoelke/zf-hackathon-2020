# This is the HYPERPARAMETER file, import in all other scripts

# Simulator
# number of states to consider in moving average
MOVING_AVG_COUNT = 100

# GeneticAlgorithm
# in an epoch a population is eveluated and updated
EPOCHS = 50
# number of nets in a generation
POPULATION_SIZE = 16
# number of best nets wich remain unchanged
NUM_SURVIVORS = 8
# prameters for mutation and crossover
MUTATION_RATE = 0.01
MUTATION_SCALE = 0.3
# how many road position we look at for each epoch
EVALUATION_REPEATS = 3
# how many steps we train from the picked road position onwards
EVALUATION_STEPS = 3000

# Data
# driving speed [m/s] between 2 and 40
VEL = [8, 20, 27]
# road type e.g. highway
K = [3,20]
#simuation interval in seconds
DT = 0.005
#record location
ROADPROFILELOCATION = '../datasets/'
LISTDATA = ['ts1_2_k_3.0.csv', 'ts1_1_k_3.0.csv', 'ts1_3_k_3.0.csv', 'ts1_4_k_3.0.csv']

# Evaluator
# where to store trained models
MODEL_PATH = '../models/2020_01_12-17_14_23/model_8.roadie'
ROAD_PROFILE_EVAL = 'pickleData/ts4_k_20.0.csv_vel8.3.pickle'
# on which velocity to evaluate
VELOCITY_eval = [27]
