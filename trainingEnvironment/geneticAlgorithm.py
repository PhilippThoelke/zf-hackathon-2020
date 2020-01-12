from environment import Simulator
from dataProcessing import ProfileManager
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn
from joblib import Parallel, delayed
import datetime
import os
from hyperparameters import *


class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.layers = [
            nn.Linear(9, 4), torch.relu,
            nn.Linear(4, 4), torch.relu,
            nn.Linear(4, 1), torch.sigmoid
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class GeneticAlgorithm:


    def __init__(self):
        self.history = []
        print('Generating population...')
        self.population = np.array([GeneticAlgorithm._get_model() for _ in range(POPULATION_SIZE)])

        print('Loading road profile...')
        self.road_profile = ProfileManager()

    def evaluate(self):
        fitness = np.zeros(len(self.population))
        for step in range(EVALUATION_REPEATS):
            # choose a random road from the CSVs
            road = self.road_profile.training_profile[np.random.randint(0, len(self.road_profile.training_profile))]
            # TO DO : choose k
            k = self.road_profile.training_profile[np.random.randint(0, len(self.road_profile.k))]
            # choose a random offset from the start of the current road
            road_offset = np.random.randint(0, len(road) - EVALUATION_STEPS)

            # evaluate all models on the current road section
            fitness += Parallel(n_jobs=-1)(delayed(GeneticAlgorithm._simulate)(model, road, road_offset, k) for model in self.population)

        # return the mean fitness for each model across multiple road sections
        return fitness / len(self.population)

    def optimization_step(self):
        fitness = self.evaluate()
        self.history.append(np.min(fitness))

        # remove worst performing models from the population
        indices = np.argsort(fitness)
        fitness = fitness[indices]
        self.population = self.population[indices]
        for i in range(NUM_SURVIVORS, len(self.population)):
            # randomly choose two parents from the surviving population
            parent1, parent2 = np.random.choice(self.population[:NUM_SURVIVORS], size=2, replace=False)
            weights1 = GeneticAlgorithm._get_weights(parent1)
            weights2 = GeneticAlgorithm._get_weights(parent2)

            # generate new model weights using crossover and mutation
            new_weights = GeneticAlgorithm._crossover(weights1, weights2)
            new_weights = GeneticAlgorithm._mutate(new_weights)

            # insert the new model into the population
            GeneticAlgorithm._set_weights(self.population[i], new_weights)

        return self.population[0]

    def save_model(model, path):
        torch.save(model.state_dict(), path)

    def load_model(path):
        model = ANN()
        model.load_state_dict(torch.load(path))
        return model

    def _simulate(model, road_profile, road_offset, k=3):
        # instantiate a new simulator
        env = Simulator(road_profile, road_offset, k)
        x = env.states[-1]
        for step in range(EVALUATION_STEPS):
            # simulate the car's behaviour and pass new i (damper current) values
            x_torch = torch.from_numpy(x.reshape((1,) + x.shape))
            x = env.next(model(x_torch)[0,0] * 2)
        return env.score()

    def _crossover(weights1, weights2):
        new_weights = []
        for var1, var2 in zip(weights1, weights2):
            assert var1.shape == var2.shape
            mask = torch.rand(size=var1.shape) < 0.5
            new_var = torch.empty(var1.shape)
            new_var[mask] = var1[mask]
            new_var[~mask] = var2[~mask]
            new_weights.append(new_var)
        return new_weights

    def _mutate(weights):
        new_weights = []
        for var in weights:
            mask = torch.rand(size=var.shape) < MUTATION_RATE
            new_var = var.clone()
            new_var[mask] += torch.randn(size=var.shape)[mask] * MUTATION_SCALE
            new_weights.append(new_var)
        return new_weights

    def _get_model():
        model = ANN()
        return model

    def _get_weights(model):
        return [layer.weight for layer in model.layers if hasattr(layer, 'weight')]

    def _set_weights(model, weights):
        index = 0
        for layer in model.layers:
            if hasattr(layer, 'weight'):
                layer.weight = nn.Parameter(weights[index])
                index += 1

if __name__ == '__main__':
    timestamp = datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
    path = '../models/' + timestamp
    os.makedirs(path)

    ga = GeneticAlgorithm()
    for epoch in range(EPOCHS):
        if epoch > 0:
            print(f'Optimization step {epoch} (fitness: {ga.history[-1]})')
        best = ga.optimization_step()
        GeneticAlgorithm.save_model(best, f'{path}/model_{epoch}.roadie')

    plt.plot(ga.history)
    plt.show()
