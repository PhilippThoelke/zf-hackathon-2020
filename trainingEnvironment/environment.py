import numpy as np
from scipy import signal

#do not change constants parameter
Mb = 500        #mass quarter body [kg]
Mt = 50        	#mass tire + suspention system [kg]
Sb = 35000      #spring constant tire to body[N/m]
St = 280000     #spring constant tire to road [N/m]
Dt = 10.02      #damper tire constant [Ns/m]
Dbound = 56000.0 #boundary damping constant [Ns/m] Verhältnis Dbound~100*Da
Da = 900        #damper constant active damping [Ns/m]
c = 560         #linear constant of active suspension [N/A]
#do not change constants parameter

# dt constant
dt = 0.005

class Simulator:

    N = 50

    def __init__(self, k=3):
        self.k = k

        self.states = []
        self.initial_state = np.zeros(9, dtype=np.float32)
        self.reset()

    def next(self, i_new):
        # perform one solving step of the differential equation using the given current i
        new_state = self.activeSuspension(*self.states[-1], i_new)
        self.states.append(new_state)
        # return the current state
        return new_state

    def reset(self):
        # return initial state
        self.states = [self.initial_state]

    def last_n_states(self):
        # poss on list of last last n states
        return np.array(self.states[-Simulator.N:])

    def score(self):
        # return the score for the current simulation (fitness)
        last = self.last_n_states()

        # extract Zb acceleration from the last N states
        Zb_dtdt = last[:,2]

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
        target = self.k * varZb_dtdt_h + varZb_dtdt
        return target

    def activeSuspension(self, Zb, Zb_dt, Zb_dtdt, Zt, Zt_dt, Zt_dtdt, i_old, Zh, Zh_dt, i):
        # old : (Zb, Zt, Zb_dt, Zt_dt, Zh, Zh_dt, i, dt):
        '''
        --- Quarter Car Suspension Model vgl. hackathon task ---
        Zb: z-position body [m]
        Zt: z-position tire [m]
        Zh: road profie [m]
        Zb_dt: velocity body in z [m/s]
        Zt_dt: velocity tire in z [m/s]
        Zh_dt: velocity road profile in z [m/s]
        ...
        x_dtdt: specific acceleration in [m/s^2]

        Tuning Parameter
        i: current of active suspension from 0 to 2 [A]
        '''

        F_push = Da * (Zb_dt - Zt_dt) - c * i
        F_bound = Dbound * (Zb_dt - Zt_dt)
        F_pull = Da * (Zb_dt - Zt_dt) + c * i

        F_D = np.max([F_push, np.min([F_bound, F_pull])])

        updated_Zb_dtdt = (-Sb * (Zb - Zt)) / Mb - F_D / Mb
        updated_Zb_dt = Zb_dt + updated_Zb_dtdt * dt
        updated_Zb = Zb + updated_Zb_dt * dt

        updated_Zt_dtdt = (-St * (Zt - Zh)) / Mt + (Sb * (Zb - Zt)) / Mt + (-Dt * (Zt_dt - Zh_dt)) / Mt + F_D / Mt
        updated_Zt_dt = Zt_dt + updated_Zt_dtdt * dt
        updated_Zt = Zt + updated_Zt_dt * dt

        # TO DO : get new road profile values (Zh, Zh_dt)
        updated_Zh = 0
        updated_Zh_dt = 0

        return np.array([updated_Zb, updated_Zb_dt, updated_Zb_dtdt, updated_Zt, updated_Zt_dt, updated_Zt_dtdt, i, updated_Zh, updated_Zh_dt], dtype=np.float32)

    def _butter_bandpass(lowcut, highcut, fs, order):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        return b, a