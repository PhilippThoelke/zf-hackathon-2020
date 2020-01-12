import numpy as np
from scipy import signal
import csv
import pandas as pd
from matplotlib import pyplot as plt
from hyperparameters import *

FILE_DIRECTORY = '../results/'
DATA_LIST = ['ts3_1_k_3.0.csv', 'ts3_2_k_3.0.csv', 'ts3_3_k_3.0.csv']
MODEL_LIST = ['grid_29.roadie', 'model_20.roadie', 'model_84.roadie']
target_csv = pd.DataFrame()

def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def calctarget(result_file):
    results = pd.read_csv(result_file)

    Zb_dtdt = results['Zb_dtdt']
    #compute bandpass 2nd order from 0.4 - 3 Hz
    b, a = butter_bandpass(0.4, 3, int(1 / DT), 2)
    zi = signal.lfilter_zi(b, a)
    z, _ = signal.lfilter(b, a, Zb_dtdt, zi=zi)

    #calculate variance alpha_1
    varZb_dtdt = np.var(z)

    #compute bandpass 2nd order from 0.4 - 3 Hz
    b, a = butter_bandpass(10, 30, int(1 / DT), 1)
    zi = signal.lfilter_zi(b, a)
    z1, _ = signal.lfilter(b, a, Zb_dtdt, zi=zi)

    #calculate variance alpha_2
    varZb_dtdt_h = np.var(z1)

    #compute T_target
    target = 3 * varZb_dtdt_h + varZb_dtdt
    #print(Zb_dtdt)
    return target


for index in range(len(MODEL_LIST)):
    for set in range (len(DATA_LIST)):
        filename = FILE_DIRECTORY + MODEL_LIST[index].split(".")[0] + "_" + DATA_LIST[set].split(".")[0] + ".csv"
        target_csv = target_csv.append({'Model': MODEL_LIST[index].split(".")[0], 'Dataset': DATA_LIST[set].split(".")[0], 'T_target': calctarget(filename)}, ignore_index = True)

print(target_csv)
target_csv.to_csv('../results/result_target.csv')
model_plot = target_csv.groupby('Model')
for model in range(len(MODEL_LIST)):
    plot_model = model_plot.get_group(MODEL_LIST[model].split(".")[0]).reset_index()
    plot_model['T_target'].plot(label = MODEL_LIST[model].split(".")[0])
plt.xticks(range(len(DATA_LIST)),DATA_LIST)
plt.xlabel('Datasets')
plt.ylabel('T_target')
plt.legend()
plt.show()
