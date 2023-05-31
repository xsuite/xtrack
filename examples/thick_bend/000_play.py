import numpy as np

import xtrack as xt
import xpart as xp

rho = 20

theta_array = np.linspace(1e-10, 2*np.pi, 100)
x_array = np.zeros_like(theta_array)
s_array = np.zeros_like(theta_array)
X0_array = np.zeros_like(theta_array)
Z0_array = np.zeros_like(theta_array)

for ii, tt in enumerate(theta_array):

    theta = tt
    bend = xt.TrueBend(length=rho*theta, h=1e-6, k0=1/rho)

    p = xp.Particles(p0c=1e9)
    bend.track(p)

    X0 = -rho*(1-np.cos(theta))
    Z0 = rho*np.sin(theta)

    X0_array[ii] = X0
    Z0_array[ii] = Z0

    x_array[ii] = p.x[0]
    s_array[ii] = p.s[0]

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.plot(X0_array, Z0_array)
plt.axis('equal')

plt.show()