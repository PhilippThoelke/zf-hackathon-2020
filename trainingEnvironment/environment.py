import numpy as np

class Simulator:

    def __init__(self, n):
        self.states.append(self.reset())
        self.n = n

    def next(self, current):
        # perform one solving step of the differential equation using the given current i
        current state = np.zeros(9)
        self.states.append(current_state)
        # return the current state
        return current_state

    def reset(self):
        # return initial state
        self.states = []
        return np.zeros(6, dtype=np.float32)

    def passs_on(self):
        # poss on list of last last n states
        return self.states[-self.n:]

    def score(self):
        # return the score for the current simulation (fitness)
        return 0

