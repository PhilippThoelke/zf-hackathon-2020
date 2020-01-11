from model import activeSuspension
import numpy as np
import csv
import os
from scipy import signal


roadProfile = 'speedBump'

# trip distance in [m]
tripMax = 125.0

# position in [m] of rectangular speed bump of 0.5m length and 0.05m hight
pSpeedBum = 10.0

# driving speed [m/s] between 2 and 30
vel = 20.0

#simuation interval in seconds
dt = 0.005

#road specific weight factor
K = 0.08 #in this artificial case below 1.0 else >1.0

intialSuspensionState = np.zeros(6) #inital suspension states at t = 0
intialSuspensionState[0] = 0 #Zb: z-position body [m]
intialSuspensionState[1] = 0 #Zt: z-position tire [m]
intialSuspensionState[2] = 0 #Zb_dt: velocity body in z [m/s]
intialSuspensionState[3] = 0 #Zt_dt: velocity tire in z [m/s]
intialSuspensionState[4] = 0 #Zb_dtdt: acceleration body in z [m/s^2]
intialSuspensionState[5] = 0 #Zt_dtdt: acceleration tire in z [m/s^2]

#get simulation time by constant speed
T = float(tripMax)/float(vel)

N = int(np.round(T/dt))
t = np.linspace(0, T, N+1)

#create road profile [m] time intervals with speed bump
profile = np.zeros(t.size)
# add speed bump of 0.25m length and 0.05m hight
profile[int((pSpeedBum/vel)/dt):int(((pSpeedBum+0.5)/vel)/dt)]=0.05

#get driving speed vector e.g for dynamic (non constant) speed
v = np.ones(t.size)*vel

#get trip at each dt
trip = []
for i in range(0,t.size):
    trip.append(np.trapz(v[0:i+1], dx=dt))

#set array of current time interval of active suspension for demo
#the real value should be computed within the solver_dampingForce()
#constant current between [0-2]
a = 1.0
C = np.ones(t.size)*a

#a possible dynamic solution which is better as the constant current
#C = np.ones(t.size)*a
#C[int(((pSpeedBum+0.01)/vel)/dt):int(((pSpeedBum+0.50)/vel)/dt)] = 2
#C[int(((pSpeedBum+0.6)/vel)/dt):int(((pSpeedBum+80.0)/vel)/dt)] = 2.0



def solver_dampingForce(intialSuspensionState, H, t, C):
    '''
    intialSuspensionState: [Zb, Zt, Zb_dt, Zt_dt] at t=0
    H: road profile [m] over time intervals
    t: time intervals

    Tuning Vector
    C: current over from 0 to 2 [A] time interval of active suspension
    '''
    N = t.size - 1             # No of time intervals
    u = np.zeros((N+1,6))      # array of [Zb, Zt, Zb_dt, Zt_dt, Zb_dtdt, Zt_dtdt] at each time interval
    u[0]=intialSuspensionState # [Zb, Zt, Zb_dt, Zt_dt] at t=0
    for n in range(0,N):
        dt = float(t[1] - t[0]) # dt
        Zh_dt = float(0)
        if n>0:
            dt = t[n] - t[n-1]
            Zh_dt = (H[n]-H[n-1])/dt
        #******
        # if no vector C with current over from 0 to 2 [A] time interval is provided
        # one may calculate the next i here based on the array u[0:n], H[0:n] and Zh_dt
        #******
        u[n+1]=activeSuspension(u[n][0], u[n][1], u[n][2], u[n][3], H[n], Zh_dt, C[n], dt)
    return u

u = solver_dampingForce(intialSuspensionState, profile, t, C)

Zb= u[:,0]
Zt= u[:,1]
Zb_dtdt= u[:,4]
Zt_dtdt= u[:,5]


def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

#compute bandpass 2nd order from 0.4 - 3 Hz
b, a = butter_bandpass(0.4, 3, int(1/dt), 2)
zi = signal.lfilter_zi(b, a)
z, _ = signal.lfilter(b, a, Zb_dtdt, zi=zi)

#calculate variance alpha_1
varZb_dtdt=np.var(z)


#compute bandpass 2nd order from 0.4 - 3 Hz
b, a = butter_bandpass(10, 30, int(1/dt), 1)
zi = signal.lfilter_zi(b, a)
z1, _ = signal.lfilter(b, a, Zb_dtdt, zi=zi)
#calculate variance alpha_2
varZb_dtdt_h=np.var(z1)

#compute T_target
target = K*varZb_dtdt_h + varZb_dtdt

#check boudning condition

#do not change constants parameter
Mb = 500        #mass quarter body [kg]
Mt = 50        	#mass tire + suspention system [kg]
#do not change constants parameter

#standard deviation of Zt_dtdt
devZt_dtdt = np.std(Zt_dtdt*Mt)

#boundary condition
F_stat_bound=(Mb+Mt)*9.81/3.0
if devZt_dtdt > (F_stat_bound):
    bc = 'failed'
else:
    bc = 'passed'

#save handin matrix for
saveArray=np.column_stack((t,profile, Zt, Zb, Zt_dtdt, Zb_dtdt, C))
saveArray[:, np.newaxis]
np.savetxt('Scoring_test.csv', saveArray, delimiter=',', fmt =['%10.3f', '%10.5f', '%10.5f', '%10.5f', '%10.5f', '%10.5f', '%10.5f'])

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

dirName = roadProfile+'vel_'+str(vel)

createFolder('plots/'+dirName)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.clf()


fig = plt.figure(figsize=(13.3,7.5))
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()

ax1.plot(t, Zb , 'b-', label="displacement Zb(t) [m]")
ax1.plot(t, Zt , 'g-', label="displacement Zt(t) [m]")
#ax1.plot(t, z , 'c-', label="Zb_dtdt(t) low")
#ax1.plot(t, z1 , 'm-', label="Zb_dtdt(t) high")
ax1.plot(t, profile, 'r-', label="road profile [m]")
nameplot = roadProfile+'vel_'+str(vel)+'/I_'+str(C[0])+'_target_'+str(target)+'_bc_'+bc
plt.title('v = '+str(vel)+' i = '+str(C[0]), y=2.08)
ax1.set_xlabel(r"time [s]")
ax1.set_ylabel(r"[m]")
ax2.set_ylim([-0.15,0.15])

tick_locations = ax1.get_xticks()
tick_locations = [i for i in tick_locations if i <= t[-1] and i >= 0]


ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks(tick_locations)
tick=(np.asarray(tick_locations)/dt).astype(int)
ax2.set_xticklabels(np.round([ trip[i] for i in tick ]))
ax2.set_xlabel(r"trip [m]")
#ax1.set_ylim([-1,1])

ax1.legend(loc="upper right")

# save plot to PNG file
plt.savefig('plots/'+nameplot+'.png', dpi=900)
plt.close(fig)
plt.clf()
#plt.show()
