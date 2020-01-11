import numpy as np

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

def activeSuspension(Zb, Zt, Zb_dt, Zt_dt, Zh, Zh_dt, i, dt):
    '''
    --- Quarter Car Suspension Model vgl. hackathon task ---
    Zb: z-position body [m]
    Zt: z-position tire [m]
    Zb_dt: velocity body in z [m/s]
    Zt_dt: velocity tire in z [m/s]
    Zh: road profie [m]
    Zh_dt: velocity road profile in z [m/s]

    Tuning Parameter
    i: current of active suspension from 0 to 2 [A]
    '''
    F_push = Da*(Zb_dt-Zt_dt) - c*i
    F_bound = Dbound*(Zb_dt-Zt_dt)
    F_pull = Da*(Zb_dt-Zt_dt) + c*i

    F_D = np.max([F_push,np.min([F_bound,F_pull])])

    updated_Zb_dtdt = (-Sb*(Zb-Zt))/Mb - F_D/Mb
    updated_Zb_dt = Zb_dt + updated_Zb_dtdt*dt
    updated_Zb = Zb + updated_Zb_dt*dt

    updated_Zt_dtdt = (-St*(Zt-Zh))/Mt + (Sb*(Zb-Zt))/Mt + (-Dt*(Zt_dt-Zh_dt))/Mt + F_D/Mt
    updated_Zt_dt = Zt_dt + updated_Zt_dtdt*dt
    updated_Zt = Zt + updated_Zt_dt*dt

    return [updated_Zb, updated_Zt, updated_Zb_dt, updated_Zt_dt, updated_Zb_dtdt, updated_Zt_dtdt ]
