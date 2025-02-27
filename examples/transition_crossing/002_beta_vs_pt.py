import xtrack as xt
import xobjects as xo
import numpy as np
from scipy.constants import c as clight

p = xt.Particles(mass0=xt.PROTON_MASS_EV, q0=1,
                 kinetic_energy0=50e6,
                 delta=np.linspace(-0.01, 0.01, 1000))
beta = p.rvv * p.beta0
beta0 = p.beta0[0]
ptau = p.ptau
delta = p.delta

deriv_beta_ptau = (1 - beta0**2)

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(2)
plt.plot(p.ptau, beta)
plt.plot(p.ptau, beta0 + deriv_beta_ptau*p.ptau)
plt.show()