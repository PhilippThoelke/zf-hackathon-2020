from environment import Simulator
from dataProcessing import ProfileManager
from tensorflow.keras import models, layers
import numpy as np
from matplotlib import pyplot as plt
from threading import Thread

class GeneticAlgorithm:

    POPULATION_SIZE = 20
    NUM_SURVIVORS = 10
    MUTATION_RATE = 0.025
    MUTATION_SCALE = 1
    EVALUATION_STEPS = 750

    def __init__(self):
        self.history = []
        print('Generating population...')
        self.population = np.array([GeneticAlgorithm._get_model() for _ in range(GeneticAlgorithm.POPULATION_SIZE)])

        print('Loading road profile...')
        self.road_profile = ProfileManager()

    def evaluate(self):
        fitness = []
        for model in self.population:
            fitness.append(self._simulate(model))
        return np.array(fitness)

    def optimization_step(self):
        fitness = self.evaluate()
        self.history.append(np.min(fitness))

        # remove worst performing models from the population
        indices = np.argsort(fitness)
        fitness = fitness[indices]
        self.population = self.population[indices]
        for i in range(GeneticAlgorithm.NUM_SURVIVORS, len(self.population)):
            # randomly choose two parents from the surviving population
            parent1, parent2 = np.random.choice(self.population[:GeneticAlgorithm.NUM_SURVIVORS], size=2, replace=False)
            weights1 = parent1.get_weights()
            weights2 = parent2.get_weights()

            # generate new model weights using crossover and mutation
            new_weights = GeneticAlgorithm._crossover(weights1, weights2)
            GeneticAlgorithm._mutate(new_weights)

            # insert the new model into the population
            self.population[i] = GeneticAlgorithm._get_model(weights=new_weights)

    def _simulate(self, model):
        env = Simulator(self.road_profile.training_profile[0])
        x = env.states[-1]
        for step in range(GeneticAlgorithm.EVALUATION_STEPS):
            x = env.next(model(x.reshape((1,) + x.shape)).numpy()[0,0])
        return env.score()

    def _crossover(weights1, weights2):
        new_weights = []
        for var1, var2 in zip(weights1, weights2):
            assert var1.shape == var2.shape
            mask = np.random.uniform(low=0, high=1, size=var1.shape) < 0.5
            new_var = np.empty(var1.shape)
            new_var[mask] = var1[mask]
            new_var[~mask] = var2[~mask]
            new_weights.append(new_var)
        return new_weights

    def _mutate(weights):
        for var in weights:
            mask = np.random.uniform(low=0, high=1, size=var.shape) < GeneticAlgorithm.MUTATION_RATE
            old = np.array(var)
            var[mask] += np.random.normal(loc=0, scale=GeneticAlgorithm.MUTATION_SCALE, size=var.shape)[mask]

    def _get_model(weights=None):
        model = models.Sequential()
        model.add(layers.Input(shape=(9,)))
        model.add(layers.Dense(units=8, activation='relu'))
        model.add(layers.Dense(units=1, activation='sigmoid'))

        if weights is not None:
            model.set_weights(weights)
        return model

if __name__ == '__main__':
    ga = GeneticAlgorithm()
    for step in range(25):
        if step > 0:
            print(f'Optimization step {step} (fitness: {ga.history[-1]})')
        ga.optimization_step()

    plt.plot(ga.history)
    plt.show()
