from solver import solver_linear_damping
import numpy as np
import csv

#simuation interval in seconds
dt = 0.005

#inital set
I = 0; V = 0
m = 50; b = 10.2

#spring model as function
def s(u):
    return 200*u

timeRecording = []
tripRecording = []
profile = []

#loading records
roadProfile = 'ts1_1_k_3.0.csv'
with open('/net/projects/scratch/winter/valid_until_31_July_2020/hackathon/datasets/'+roadProfile) as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        timeRecording.append(float(row[0]))
        tripRecording.append(float(row[1]))
        profile.append(float(row[2]))


#get simulation time by constant speed
T = timeRecording[-1]

N = int(np.round(T/dt))
t = np.linspace(0, T, N+1)

#get driving speed vector from recoding trip -> differentiate trip
dydx = np.diff(tripRecording)/np.diff(timeRecording)
v = np.interp(t, np.asarray(timeRecording)[1:], dydx)

#get trip at each dt
trip = []
for i in range(0,t.size):
    trip.append(np.trapz(v[0:i+1], dx=dt))

#get the road profile by the tripRecording
profile = np.interp(trip, tripRecording, profile)


def acceleration(h, x, v):
    """Compute 2nd-order derivative of h."""
    d2h = np.zeros(h.size)
    dx = x[1] - x[0]
    # Method: standard finite difference aproximation -> slow but can easily be altered
    for i in range(1, h.size-1, 1):
        dx = x[i] - x[i-1]
        if dx > 0.075: # should be dx > 0 but the resampling of the dataset create noise
            d2h[i] = (h[i-1] - 2*h[i] + h[i+1])/dx**2
        else:
            d2h[i] = 0
    # Extraplolate end values from first interior value
    d2h[0] = d2h[1]
    d2h[-1] = d2h[-2]
    a = d2h*v**2
    return a

a = acceleration(profile, np.asarray(trip), np.asarray(v))
F = -m*a

u = solver_linear_damping(I, V, m, b, s, F, t)



import matplotlib.pyplot as plt
plt.clf()


fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()

ax1.plot(t, u , 'b-', label="displacement u(t)")
ax1.plot(t, profile , 'r-', label="road profile")
ax1.set_xlabel(r"time [s]")
ax1.set_ylabel(r"[m]")


tick_locations = ax1.get_xticks()
tick_locations = [i for i in tick_locations if i <= t[-1] and i >= 0]

ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks(tick_locations)
tick=(np.asarray(tick_locations)/dt).astype(int)
ax2.set_xticklabels(np.round([ trip[i] for i in tick ]))
ax2.set_xlabel(r"trip [m]")
ax1.legend(loc="upper right")

plt.savefig('sDamperProfileRecordingRecVel.pdf')
# save plot to PDF file
plt.savefig('sDamperProfileRecordingRecVel.png')
# save plot to PNG file
plt.show()
