from tensorflow.keras import models, layers
from environment import Simulator

class GeneticAlgorithm:

    POPULATION_SIZE = 5

    def __init__(self):
        self.population = [GeneticAlgorithm._get_model() for _ in range(GeneticAlgorithm.POPULATION_SIZE)]
        self.env = Simulator()

    def _get_model():
        model = models.Sequential()
        model.add(layers.Input(shape=(6,)))
        model.add(layers.Dense(units=8, activation='relu'))
        model.add(layers.Dense(units=1, activation='sigmoid'))
        return model
