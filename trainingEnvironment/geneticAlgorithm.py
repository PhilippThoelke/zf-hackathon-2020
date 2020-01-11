from environment import Simulator
from dataProcessing import ProfileManager
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn
from joblib import Parallel, delayed

class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.layers = [
            nn.Linear(9, 8), torch.relu,
            nn.Linear(8, 1), torch.sigmoid
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class GeneticAlgorithm:

    POPULATION_SIZE = 32
    NUM_SURVIVORS = 16
    MUTATION_RATE = 0.05
    MUTATION_SCALE = 1
    EVALUATION_STEPS = 750

    def __init__(self):
        self.history = []
        print('Generating population...')
        self.population = np.array([GeneticAlgorithm._get_model() for _ in range(GeneticAlgorithm.POPULATION_SIZE)])

        print('Loading road profile...')
        self.road_profile = ProfileManager()

    def evaluate(self):
        fitness = Parallel(n_jobs=8)(delayed(GeneticAlgorithm._simulate)(model, self.road_profile.training_profile[0]) for model in self.population)
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
            weights1 = GeneticAlgorithm._get_weights(parent1)
            weights2 = GeneticAlgorithm._get_weights(parent2)

            # generate new model weights using crossover and mutation
            new_weights = GeneticAlgorithm._crossover(weights1, weights2)
            new_weights = GeneticAlgorithm._mutate(new_weights)

            # insert the new model into the population
            GeneticAlgorithm._set_weights(self.population[i], new_weights)

    def _simulate(model, road_profile):
        env = Simulator(road_profile)
        x = env.states[-1]
        for step in range(GeneticAlgorithm.EVALUATION_STEPS):
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
            mask = torch.rand(size=var.shape) < GeneticAlgorithm.MUTATION_RATE
            new_var = var.clone()
            new_var[mask] += torch.randn(size=var.shape)[mask] * GeneticAlgorithm.MUTATION_SCALE
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
    ga = GeneticAlgorithm()
    for step in range(100):
        if step > 0:
            print(f'Optimization step {step} (fitness: {ga.history[-1]})')
        ga.optimization_step()

    plt.plot(ga.history)
    plt.show()
