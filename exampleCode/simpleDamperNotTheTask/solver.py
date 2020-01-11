import numpy as np

def solver_linear_damping(I, V, m, b, s, F, t):
    N = t.size - 1              # No of time intervals
    dt = t[1] - t[0]            # Time step
    u = np.zeros(N+1)           # Result array
    b = float(b); m = float(m)  # Avoid integer division

    # Convert F to array
    if callable(F):
        F = F(t)
    elif isinstance(F, (list,tuple,np.ndarray)):
        F = np.asarray(F)
    else:
        raise TypeError(
            'F must be function or array, not %s' % type(F))
    if not isinstance(V, (float,int)):
        raise TypeError('V must be float or int, not %s' % type(V))
    if not isinstance(I, (float,int)):
        raise TypeError('I must be float or int, not %s' % type(I))
    u[0] = I
    u[1] = u[0] + dt*V + dt**2/(2*m)*(-b*V - s(u[0]) + F[0])

    for n in range(1,N):
        u[n+1] = 1./(m + b*dt/2)*(2*m*u[n] + \
                 (b*dt/2 - m)*u[n-1] + dt**2*(F[n] - s(u[n])))
    return u
