from solver import solver_linear_damping
import numpy as np

def s(u):
    return 0.1*u

T = 10*np.pi
# simulate for t in [0,T]
dt = 0.2
N = int(np.round(T/dt))
t = np.linspace(0, T, N+1)
F = np.zeros(t.size)
I = 1; V = 0
m = 2; b = 0.2

u = solver_linear_damping(I, V, m, b, s, F, t)

from matplotlib.pyplot import *
plot(t, u)
savefig('simpleDamper.pdf')
# save plot to PDF file
savefig('simpleDamper.png')
# save plot to PNG file
show()
