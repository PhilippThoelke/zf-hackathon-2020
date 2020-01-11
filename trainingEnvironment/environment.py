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
    def __init__(self, road_profile, road_offset=0, k=3):
        self.current_road_position = road_offset + 1
        self.road_profile = road_profile
        self.k = k

        self.states = []
        self.reset()

    def next(self, i_new):
        # perform one solving step of the differential equation using the given current i
        new_state = self.active_suspension(*self.states[-1], i_new)
        self.current_road_position += 1
        self.states.append(new_state)
        # return the current state
        return new_state

    def reset(self):
        # return initial state
        self.states = [np.zeros(9, dtype=np.float32)]

    def get_states(self):
        return np.array(self.states)

    def t_target(self):
        last = self.get_states()

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

    def constraint_satisfied(self):
        last = self.get_states()

        # extract Zt acceleration from the last N states
        Zt_dtdt = last[:,5]

        #standard deviation of Zt_dtdt
        devZt_dtdt = np.std(Zt_dtdt * Mt)

        #boundary condition
        F_stat_bound = (Mb + Mt) * 9.81 / 3.0
        return devZt_dtdt <= F_stat_bound

    def score(self):
        # return the score for the current simulation (fitness)
        target = self.t_target()
        satisfied = self.constraint_satisfied()

        if not satisfied:
            target *= 10
        return target

    def active_suspension(self, Zb, Zb_dt, Zb_dtdt, Zt, Zt_dt, Zt_dtdt, i_old, Zh, Zh_dt, i):
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

        # get new road profile values (Zh, Zh_dt)
        pos = self.current_road_position
        updated_Zh = self.road_profile[pos]
        updated_Zh_dt = (self.road_profile[pos] - self.road_profile[pos - 1]) / dt

        return np.array([updated_Zb, updated_Zb_dt, updated_Zb_dtdt, updated_Zt, updated_Zt_dt, updated_Zt_dtdt, i, updated_Zh, updated_Zh_dt], dtype=np.float32)

    def _butter_bandpass(lowcut, highcut, fs, order):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        return b, a