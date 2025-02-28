import xtrack as xt
import xpart as xp
from xpart.longitudinal.rf_bucket import RFBucket

import numpy as np
from scipy.constants import c as clight
from scipy.constants import e as qe
import matplotlib.pyplot as plt

kinetic_energy0 = 100e6 # eV
gamma_transition = 1.3
momentum_compaction_factor = 1 / gamma_transition**2

particle_ref = xt.Particles(kinetic_energy0=kinetic_energy0,
                            mass0=xt.PROTON_MASS_EV)

circumference = 1000.
t_rev = circumference / (particle_ref.beta0[0] * clight)
f_rev = 1 / t_rev

energy_ref_increment =  0 #0.001e6 # eV


h_rf = 20

f_rf = h_rf * f_rev
v_rf = 1.5e3
lag_rf = 50

# Compute momentum increment using auxiliary particle
p_ref_aux = xt.Particles(kinetic_energy0=kinetic_energy0 + energy_ref_increment,
                         mass0=particle_ref.mass0)
dp0c_eV = p_ref_aux.p0c[0] - particle_ref.p0c[0]
dp0c_J = dp0c_eV * qe
dp0_si = dp0c_J / clight

otm = xt.LineSegmentMap(
    betx=1., bety=1,
    qx=6.3, qy=6.4,
    momentum_compaction_factor=momentum_compaction_factor,
    longitudinal_mode="nonlinear",
    voltage_rf=v_rf,
    frequency_rf=f_rf,
    lag_rf=lag_rf,
    length=circumference,
    energy_ref_increment=energy_ref_increment
)

line = xt.Line(elements=[otm], particle_ref=particle_ref)

tw = line.twiss()

delta_test = np.linspace(0, 1.5e-3, 20)
p = line.build_particles(delta=delta_test)

line.track(p, turn_by_turn_monitor=True, num_turns=1000)
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
    p_increment=dp0_si,
    harmonic_list=[h_rf],
    voltage_list=[v_rf],
    phi_offset_list=[np.deg2rad(lag_rf)],
)

z_separatrix = np.linspace(-30, 30, 1000)
delta_separatrix = rfb.separatrix(z_separatrix)

p_gauss, matcher = xp.generate_matched_gaussian_bunch(
    line=line,
    num_particles=1000,
    nemitt_x=2.5e-6,
    nemitt_y=2.5e-6,
    sigma_z=5,
    return_matcher=True)

plt.close('all')
plt.figure(1)
plt.plot(p_gauss.zeta, p_gauss.delta, '.', color='k', alpha=0.5)
plt.plot(mon.zeta.T, mon.delta.T, color='C0')
plt.plot(z_separatrix, delta_separatrix, color='C1')
plt.plot(z_separatrix, -delta_separatrix, color='C1')
plt.xlabel('zeta [m]')
plt.ylabel('delta')

# Check the force
z_test = np.linspace(-30, 30, 1000)
force = matcher.rfbucket.total_force(z_test)
hamiltonian = rfb.hamiltonian(z_test, 0)

plt.figure(2)
ax1 = plt.subplot(2,1,1)
ax1.plot(z_test, force)
ax1.set_ylabel(r'F($\zeta$)')
ax2 = plt.subplot(2,1,2, sharex=ax1)
ax2.plot(z_test, matcher.rfbucket.hamiltonian(z_test, 0))
ax2.set_xlabel(r'$\zeta$ [m]')
ax2.set_ylabel(r'$H(\zeta, 0)$')

# Check hamiltonian on the delta axis
delta_test = np.linspace(-1e-2, 1e-2, 1000)
plt.figure(3)
plt.plot(delta_test, matcher.rfbucket.hamiltonian(0, delta_test))
plt.plot(delta_test, matcher.psi_object.H(0, delta_test))
plt.plot(delta_test, matcher.rfbucket.hamiltonian(0, delta_test, make_convex=True), '--')
plt.xlabel(r'$\delta$')
plt.ylabel(r'$H(0, \delta)$')

plt.show()