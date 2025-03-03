import xtrack as xt
import xobjects as xo
import numpy as np
from scipy.constants import c as clight

p = xt.Particles(mass0=xt.PROTON_MASS_EV, q0=1,
                 kinetic_energy0=50e6, delta=0.1)

beta0 = p.beta0[0]
gamma0 = p.gamma0[0]
mass0_ev = p.mass0
energy0 = p.energy0[0]
energy = p.energy[0]
p0c = p.p0c[0]

delta = p.delta[0]
ptau = p.ptau[0]
pzeta = p.pzeta[0]

beta = beta0 * p.rvv[0]
gamma = energy / energy0 * gamma0

xo.assert_allclose(gamma0, 1 / np.sqrt(1 - beta0**2), rtol=0, atol=1e-14)
xo.assert_allclose(gamma, 1 / np.sqrt(1 - beta**2), rtol=0, atol=1e-14)
xo.assert_allclose(energy, mass0_ev * gamma, rtol=0, atol=1e-6) # 1e-6 eV
xo.assert_allclose(energy0, mass0_ev * gamma0, rtol=0, atol=1e-6) # 1e-6 eV

Pc = p0c * (1 + delta)
xo.assert_allclose(Pc, mass0_ev * gamma * beta, rtol=0, atol=1e-6) # 1e-6 eV

# Definitions delta/ptau/pzeta
xo.assert_allclose(delta, (Pc - p0c) / p0c, rtol=0, atol=1e-14)
xo.assert_allclose(ptau, (energy - energy0) / energy0 / beta0, rtol=0, atol=1e-14)
xo.assert_allclose(pzeta,(energy - energy0) / energy0 / beta0**2, rtol=0, atol=1e-14)
xo.assert_allclose(pzeta,(energy - energy0) / p0c / beta0, rtol=0, atol=1e-14)

# Conversions
xo.assert_allclose(delta, np.sqrt(ptau**2 + 2*ptau/beta0 + 1) - 1, rtol=0, atol=1e-14)
xo.assert_allclose(delta, np.sqrt(beta0**2 * pzeta**2 + 2*pzeta + 1) - 1, rtol=0, atol=1e-14)
xo.assert_allclose(delta, beta * ptau + (beta - beta0)/beta0, rtol=0, atol=1e-14)
xo.assert_allclose(delta, beta * beta0 * pzeta + (beta - beta0)/beta0, rtol=0, atol=1e-14)

# More relations from the physics manual
xo.assert_allclose(gamma, gamma0 * (1 + beta0 * ptau), rtol=0, atol=1e-14)
xo.assert_allclose(beta, np.sqrt(1 - (1 - beta0**2)/(1 + beta0 * ptau)**2), rtol=0, atol=1e-14)

# Check relations for small energy deviations
p_small = xt.Particles(mass0=xt.PROTON_MASS_EV, q0=1,
                       kinetic_energy0=50e6, delta=1e-4)
delta_small = p_small.delta[0]
ptau_small = p_small.ptau[0]
pzeta_small = p_small.pzeta[0]
beta_small = beta0 * p_small.rvv[0]

xo.assert_allclose(delta_small, ptau_small/beta0, rtol=0, atol=1e-8)
xo.assert_allclose(delta_small, pzeta_small, rtol=0, atol=1e-8)
xo.assert_allclose(beta_small, beta0 + (1 - beta0**2) * ptau_small, rtol=0, atol=1e-8)