import numpy as np

import xtrack as xt
import xpart as xp


theta_array = np.linspace(0, 2*np.pi, 100)
x_array = np.zeros_like(theta_array)
s_array = np.zeros_like(theta_array)
for ii, tt in enumerate(theta_array):

    rho = 20
    theta = 0.1
    bend = xt.TrueBend(length=rho*theta, h=1e-10, k0=1/rho)

    p = xp.Particles(p0c=1e9)
    bend.track(p)

    x_array[ii] = p.x[0]
    s_array[ii] = p.s[0]