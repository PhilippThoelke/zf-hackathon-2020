import numpy as np

class Simulator:
    def __init__(self):
        pass

    def next(self, current):
        # perform one solving step of the differential equation using the given current i

        # return the current state
        return np.zeros(6, dtype=np.float32)

    def reset(self):
        # return initial state
        return np.zeros(6, dtype=np.float32)

    def score(self):
        # return the score for the current simulation (fitness)
        return 0