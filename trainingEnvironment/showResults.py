import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from environment import Simulator
from scipy import signal

dt = 0.005
k = 3

Mb = 500        #mass quarter body [kg]
Mt = 50        	#mass tire + suspention system [kg]

def t_target(history):
    # extract Zb acceleration from the last N states
    print(np.array(history).shape)
    Zb_dtdt = np.array(history)[:,5]

    #compute bandpass 2nd order from 0.4 - 3 Hz
    b, a = Simulator._butter_bandpass(0.4, 3, int(1 / dt), 2)
    zi = signal.lfilter_zi(b, a)
    z, _ = signal.lfilter(b, a, Zb_dtdt, zi=zi)

    #calculate variance alpha_1
    varZb_dtdt = np.var(z)

    #compute bandpass 2nd order from 0.4 - 3 Hz
    b, a = Simulator._butter_bandpass(10, 30, int(1 / dt), 1)
    zi = signal.lfilter_zi(b, a)
    z1, _ = signal.lfilter(b, a, Zb_dtdt, zi=zi)

    #calculate variance alpha_2
    varZb_dtdt_h = np.var(z1)

    #compute T_target
    target = k * varZb_dtdt_h + varZb_dtdt
    return target

def constraint_satisfied(history):
    # extract Zt acceleration from the last N states
    Zt_dtdt = np.array(history)[:,4]

    #standard deviation of Zt_dtdt
    devZt_dtdt = np.std(Zt_dtdt * Mt)

    #boundary condition
    F_stat_bound = (Mb + Mt) * 9.81 / 3.0
    return devZt_dtdt <= F_stat_bound

df = pd.read_csv('result.csv')

print('T_target:', t_target(df.values))
print('Constraint satisfied:', constraint_satisfied(df.values))

df['i'].plot()
plt.show()