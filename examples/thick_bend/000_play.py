import numpy as np

import xtrack as xt
import xpart as xp

rho = 20

theta_array = np.linspace(1e-10, 2*np.pi, 100)
x_array = np.zeros_like(theta_array)
s_array = np.zeros_like(theta_array)
X0_array = np.zeros_like(theta_array)
Z0_array = np.zeros_like(theta_array)

X_array = np.zeros_like(theta_array)
Z_array = np.zeros_like(theta_array)

for ii, tt in enumerate(theta_array):

    theta = tt
    bend = xt.TrueBend(length=rho*theta, h=1/rho, k0=1/rho)

    p = xp.Particles(p0c=1e9)
    p.x += 10
    p.px += 0.1
    p.delta = 0.1
    bend.track(p)

    X0 = -rho*(1-np.cos(theta))
    Z0 = rho*np.sin(theta)

    ex_X = np.cos(theta)
    ex_Z = np.sin(theta)

    X0_array[ii] = X0
    Z0_array[ii] = Z0

    x_array[ii] = p.x[0]
    s_array[ii] = p.s[0]

    X_array[ii] = X0 + p.x[0]*ex_X
    Z_array[ii] = Z0 + p.x[0]*ex_Z

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.plot(X0_array, Z0_array, 'b')
plt.plot(X_array, Z_array, 'r')
plt.axis('equal')

plt.show()