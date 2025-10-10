import xtrack as xt
import numpy as np

from scipy.constants import e as qe
from scipy.constants import c as clight

p_ref1 = xt.Particles(
    mass0=xt.PROTON_MASS_EV,
    q0=2,
    gamma0=1.2)

p_ref2 = xt.Particles(
    mass0=2*xt.PROTON_MASS_EV,
    q0=3,
    gamma0=1.5)

# Build a particle referred to reference 1
p1 = p_ref1.copy()
p1.x = 1e-3
p1.y = 2e-3
p1.delta = 0.5
P_p1 = (1 + p1.delta) * p1.p0c

# Build the same particle referred to reference 2
p1_ref2 = p_ref2.copy()
p1_ref2_mass_ratio = p1.mass0 / p1_ref2.mass0
p1_ref2_charge_ratio = p1.q0 / p1_ref2.q0

p1_ref2.x = p1.x
p1_ref2.y = p1.y
p1_ref2.zeta = p1.zeta / p1.beta0 * p1_ref2.beta0
p1_ref2.charge_ratio = p1_ref2_charge_ratio
p1_ref2.chi = p1_ref2_charge_ratio / p1_ref2_mass_ratio
p1_ref2.delta = P_p1 / p1_ref2_mass_ratio / p1_ref2.p0c - 1

assert np.isclose(p1_ref2.mass, p1.mass, atol=0, rtol=1e-14)
assert np.isclose(p1_ref2.charge, p1.charge, atol=0, rtol=1e-14)
assert np.isclose(p1_ref2.energy, p1.energy, atol=0, rtol=1e-14)
assert np.isclose(p1_ref2.rvv * p1_ref2.beta0, p1.rvv * p1.beta0, atol=0, rtol=1e-14)

assert np.isclose(p1.rpp, 1 / (1 + p1.delta), atol=0, rtol=1e-14)
assert np.isclose(p1_ref2.rpp, 1 / (1 + p1_ref2.delta), atol=0, rtol=1e-14)

p1c = p1.p0c / p1.rpp * p1.mass_ratio
p1c_ref2 = p1_ref2.p0c / p1_ref2.rpp * p1_ref2.mass_ratio
assert np.isclose(p1c, p1c_ref2, atol=0, rtol=1e-14)

model = 'bend-kick-bend'
model = 'rot-kick-rot'
model = 'expanded'
L_bend = 1.
S_Tm2 = 0.01
Ss_Tm2 = -0.05

P0_J_ref1 = p_ref1.p0c[0] / clight * qe
k2_ref1 = S_Tm2 * qe * p_ref1.charge[0] / P0_J_ref1
k2s_ref1 = Ss_Tm2 * qe * p_ref1.charge[0] / P0_J_ref1

P0_J_ref2 = p_ref2.p0c[0] / clight * qe
k2_ref2 = S_Tm2 * qe * p_ref2.charge[0] / P0_J_ref2
k2s_ref2 = Ss_Tm2 * qe * p_ref2.charge[0] / P0_J_ref2

n_slices = 10

sext_ref1 = xt.Sextupole(k2=k2_ref1, k2s=k2s_ref1, length=L_bend/n_slices)
sext_ref2 = xt.Sextupole(k2=k2_ref2, k2s=k2s_ref2, length=L_bend/n_slices)

ele_ref1 = []
for ii in range(n_slices):
    ele_ref1.append(xt.Drift(length=L_bend/n_slices/2))
    ele_ref1.append(sext_ref1)
    ele_ref1.append(xt.Drift(length=L_bend/n_slices/2))

ele_ref2 = []
for ii in range(n_slices):
    ele_ref2.append(xt.Drift(length=L_bend/n_slices/2))
    ele_ref2.append(sext_ref2)
    ele_ref2.append(xt.Drift(length=L_bend/n_slices/2))

line_ref1 = xt.Line(elements=ele_ref1)
line_ref2 = xt.Line(elements=ele_ref2)

line_ref1.append_element(element=xt.Marker(), name='endmarker')
line_ref2.append_element(element=xt.Marker(), name='endmarker')

line_ref1.configure_bend_model(core=model)
line_ref2.configure_bend_model(core=model)

line_ref1.build_tracker()
line_ref2.build_tracker()

line_ref1.track(p1, ele_start=0, ele_stop='endmarker')
line_ref2.track(p1_ref2, ele_start=0, ele_stop='endmarker')

assert np.isclose(p1.x, p1_ref2.x, atol=0, rtol=1e-10)
assert np.isclose(p1.y, p1_ref2.y, atol=0, rtol=1e-10)
assert np.isclose(p1_ref2.mass, p1.mass, atol=0, rtol=1e-14)
assert np.isclose(p1_ref2.charge, p1.charge, atol=0, rtol=1e-14)
assert np.isclose(p1_ref2.energy, p1.energy, atol=0, rtol=1e-14)
assert np.isclose(p1_ref2.rvv * p1_ref2.beta0, p1.rvv * p1.beta0, atol=0, rtol=1e-14)

# Check absolute time of arrival
t0_ref1 = p1.s / (p1.beta0 * clight)           # Absolute reference time of arrival
t0_ref2 = p1_ref2.s / (p1_ref2.beta0 * clight) # Absolute reference time of arrival
dt_ref1 = -p1.zeta / (p1.beta0 * clight)           # Arrival time relative to reference
dt_ref2 = -p1_ref2.zeta / (p1_ref2.beta0 * clight) # Arrival time relative to reference

assert np.isclose(t0_ref1 + dt_ref1, t0_ref2 + dt_ref2, atol=1e-11, rtol=0)
