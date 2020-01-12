import pandas as pd
from matplotlib import pyplot as plt
from hyperparameters import *
import numpy as np
from environment import Simulator
from scipy import signal
from evaluator import t_target, constraint_satisfied

FILE = 'result_i_0.csv'

PLOT_OFFSET = 300
PLOT_LENGTH = 3000

df = pd.read_csv(FILE)

print(df)

print('T_target:', t_target(df.values))
print('Constraint satisfied:', constraint_satisfied(df.values))

df.iloc[PLOT_OFFSET:PLOT_OFFSET+PLOT_LENGTH,[1,6]].plot()
plt.show()
