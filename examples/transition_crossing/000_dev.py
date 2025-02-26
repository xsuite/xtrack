import xtrack as xt
import xpart as xp
from xpart.longitudinal.rf_bucket import RFBucket

import numpy as np
from scipy.constants import c as clight
from scipy.constants import e as qe
import matplotlib.pyplot as plt

kinetic_energy0 = 200e6 # eV
gamma_transition = 1.3
momentum_compaction_factor = 1 / gamma_transition**2

particle_ref = xt.Particles(kinetic_energy0=kinetic_energy0,
                            mass0=xt.PROTON_MASS_EV)

circumference = 1000.
t_rev = circumference / (particle_ref.beta0[0] * clight)
f_rev = 1 / t_rev

h_rf = 20

f_rf = h_rf * f_rev
v_rf = 1.5e3
lag_rf = 0.

otm = xt.LineSegmentMap(
    betx=1., bety=1,
    qx=6.3, qy=6.4,
    momentum_compaction_factor=momentum_compaction_factor,
    longitudinal_mode="nonlinear",
    voltage_rf=0*v_rf,
    frequency_rf=f_rf,
    lag_rf=lag_rf,
    length=100.)

cav = xt.Cavity(
    voltage=v_rf,
    frequency=f_rf,
    lag=lag_rf,
)

line = xt.Line(elements=[otm, cav], particle_ref=particle_ref)

tw = line.twiss()

delta_test = np.linspace(0, 5e-3, 20)
p = line.build_particles(delta=delta_test)

line.track(p, turn_by_turn_monitor=True, num_turns=5000)
mon = line.record_last_track

mass0_ev = particle_ref.mass0
mass0_j = mass0_ev * qe
mass_kg = mass0_j / clight**2
rfb = RFBucket(
    circumference=circumference,
    gamma=tw.gamma0,
    mass_kg=mass_kg,
    charge_coulomb=particle_ref.q0 * qe,
    alpha_array=[momentum_compaction_factor],
    p_increment=0,
    harmonic_list=[h_rf],
    voltage_list=[v_rf],
    phi_offset_list=[np.rad2deg(lag_rf)],
)

z_separatrix = np.linspace(-30, 30, 1000)
delta_separatrix = rfb.separatrix(z_separatrix)

p_gauss = xp.generate_matched_gaussian_bunch(
    line=line,
    num_particles=1000,
    nemitt_x=2.5e-6,
    nemitt_y=2.5e-6,
    sigma_z=10)

plt.close('all')
plt.figure(1)
plt.plot(p_gauss.zeta, p_gauss.delta, '.', color='k', alpha=0.5)
plt.plot(mon.zeta.T, mon.delta.T, color='C0')
plt.plot(z_separatrix, delta_separatrix/tw.beta0**2, color='C1')
plt.xlabel('zeta [m]')
plt.ylabel('delta')

plt.show()