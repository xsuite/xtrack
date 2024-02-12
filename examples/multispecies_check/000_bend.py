import xtrack as xt
import numpy as np

from scipy.constants import e as qe
from scipy.constants import c as clight

particle_ref1 = xt.Particles(
    mass0=xt.PROTON_MASS_EV,
    q0=2,
    gamma0=1.2)

particle_ref2 = xt.Particles(
    mass0=2*xt.PROTON_MASS_EV,
    q0=3,
    gamma0=1.5)

# Build a particle referred to reference 1
p1 = particle_ref1.copy()
p1.x = 1e-3
p1.y = 2e-3
p1.zeta = 1e-2
p1.delta = 0.5
P_p1 = (1 + p1.delta) * p1.p0c

# Build the same particle referred to reference 2
p1_ref2 = particle_ref2.copy()
p1_ref2_mass_ratio = p1.mass0 / p1_ref2.mass0
p1_ref2_charge_ratio = p1.q0 / p1_ref2.q0

p1_ref2.x = p1.x
p1_ref2.y = p1.y
p1_ref2.zeta = p1.zeta
p1_ref2.charge_ratio = p1_ref2_charge_ratio
p1_ref2.chi = p1_ref2_charge_ratio / p1_ref2_mass_ratio
p1_ref2.delta = P_p1 / p1_ref2_mass_ratio / p1_ref2.p0c - 1

assert np.isclose(p1_ref2.mass, p1.mass, atol=0, rtol=1e-14)
assert np.isclose(p1_ref2.charge, p1.charge, atol=0, rtol=1e-14)
assert np.isclose(p1_ref2.energy, p1.energy, atol=0, rtol=1e-14)



prrrrr




L_bend = 1.
B_T = 2

p0c = 5e9

P0_J = p0c[0] / clight * qe
h_bend = B_T * qe / P0_J
theta_bend = h_bend * L_bend



