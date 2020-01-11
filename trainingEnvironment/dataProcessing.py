import numpy as np
import csv
import os
from scipy import signal
from matplotlib import pyplot as plt

# HYPERPARAMETER
# driving speed [m/s] between 2 and 40
VEL = 27.0
#simuation interval in seconds
DT = 0.005
#record location
ROADPROFILELOCATION = '../datasets/'
LISTDATA = ['ts1_2_k_3.0.csv']

class ProfileManager:

    def __init__(self, split=False):

        #loading records
        #list of road specific weight factors
        self.K = []
        for roadProfile in LISTDATA:
            tmp = roadProfile.split("_k_")
            tmp = tmp[1]
            k = float(tmp[:-4])
            self.K.append(k)

        self.training_profile = []
        self.validation_profile = []

        # if no split -> all data to training data
        if (not split):
            self.training_profile = [self.csv_to_profile(roadProfile) for roadProfile in LISTDATA]

        # TODO: else calls split function

    def csv_to_profile(self, roadProfile):
        ## get road profile for a constant speed
        timeRecording = []
        tripRecording = []
        profile = []

        with open(ROADPROFILELOCATION+roadProfile) as csvDataFile:
            csvReader = csv.reader(csvDataFile)
            for row in csvReader:
                timeRecording.append(float(row[0]))
                tripRecording.append(float(row[1]))
                profile.append(float(row[2]))

        #get simulation time by constant speed
        T = float(tripRecording[-1])/float(VEL)

        N = int(np.round(T/DT))
        t = np.linspace(0, T, N+1)

        #get driving speed vector e.g for dynamic (non constant) speed
        v = np.ones(t.size)*VEL

        #get trip at each dt
        trip = []
        for i in range(0, t.size):
            trip.append(np.trapz(v[0:i+1], dx=DT))

        #get the road profile by the tripRecording
        profile = np.interp(trip, tripRecording, profile)

        return profile

    # TODO: splitting in train and validation data
    # TODO: feed profile to simulator

if __name__ == '__main__':
    profi = ProfileManager()
    plt.plot(profi.training_profile[0])
    plt.show()
