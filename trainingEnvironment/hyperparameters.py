# This is the HYPERPARAMETER file, import in all other scripts


# GeneticAlgorithm
# in an epoch a population is eveluated and updated
EPOCHS = 250
# number of nets in a generation
POPULATION_SIZE = 32
# number of best nets wich remain unchanged
NUM_SURVIVORS = 8
# prameters for mutation and crossover
MUTATION_RATE = 0.01
MUTATION_SCALE = 0.5
# how many road position we look at for each epoch
EVALUATION_REPEATS = 5
# how many steps we train from the picked road position onwards
EVALUATION_STEPS = 1500

# Data
# driving speed [m/s] between 2 and 40
VEL = [8, 20, 27]
# road type e.g. highway
K = [3,20]
#simuation interval in seconds
DT = 0.005
#record location
ROADPROFILELOCATION = 'datasets/'
LISTDATA = ['ts1_2_k_3.0.csv', 'ts1_1_k_3.0.csv', 'ts1_3_k_3.0.csv', 'ts1_4_k_3.0.csv']

# Evaluator
# where to store trained models
MODEL_PATH = 'models/2020_01_11-18_33_27/model_49.roadie'
ROAD_PROFILE_EVAL = 'ts3_1_k_3.0.csv'
# on which velocity to evaluate
VELOCITY_eval = [8]
