import xtrack as xt
import xobjects as xo
import numpy as np

p = xt.Particles(mass0=xt.PROTON_MASS_EV, q0=1,
                 kinetic_energy0=50e6, delta=0.1)

beta0 = p.beta0[0]
gamma0 = p.gamma0[0]
mass0_ev = p.mass0
energy0 = p.energy0[0]
energy = p.energy[0]

beta = beta0 * p.rvv[0]
gamma = energy / energy0 * gamma0

xo.assert_allclose(gamma0, 1 / np.sqrt(1 - beta0**2), rtol=0, atol=1e-14)
xo.assert_allclose(gamma, 1 / np.sqrt(1 - beta**2), rtol=0, atol=1e-14)

