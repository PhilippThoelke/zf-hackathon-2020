import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('HalfTimeScoring_tf_3_1_vel27.0.csv')
df['i'].plot()
plt.show()