from solver import solver_linear_damping
import numpy as np



X = 1000 # trip meters
# simulate for x [0,X] meters
Vel = 2 # driving speed as meter/second
#simuation interval in seconds
dt = 0.1

#inital set
I = 1; V = 0
m = 2; b = 0.2

#spring model as function
def s(u):
    return 2*u

#get simulation time by constant speed
T = float(X)/float(Vel)

N = int(np.round(T/dt))
t = np.linspace(0, T, N+1)

#get driving speed vector e.g for dynamic (non constant) speed
v = np.ones(t.size)*Vel

#get trip at each dt
trip = []
for i in range(0,t.size):
    trip.append(np.trapz(v[0:i+1], dx=dt))


#define road profile by function
def H(x):
    prof = A*np.sin(0.03*np.pi*x)+np.sin(0.02*np.pi*x+np.pi/(np.random.rand(1)*3))
    return prof

#amplitue of A in h(x)
A = 0.25

profile = H(np.asarray(trip))

def acceleration(h, x, v):
    """Compute 2nd-order derivative of h."""
    d2h = np.zeros(h.size)
    dx = x[1] - x[0]
    # Method: standard finite difference aproximation -> slow but can easily be altered
    #for i in range(1, h.size-1, 1):
    #    d2h[i] = (h[i-1] - 2*h[i] + h[i+1])/dx**2
    # Method: vectorized difference aproximation
    d2h[1:-1] = (h[:-2] - 2*h[1:-1] + h[2:])/dx**2
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

plt.savefig('sDamperProfile.pdf')
# save plot to PDF file
plt.savefig('sDamperProfile.png')
# save plot to PNG file
plt.show()
