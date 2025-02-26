import xtrack as xt
import numpy as np
from scipy.constants import c as clight
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

otm = xt.LineSegmentMap(
    betx=1., bety=1,
    qx=6.3, qy=6.4,
    momentum_compaction_factor=momentum_compaction_factor,
    longitudinal_mode="nonlinear",
    voltage_rf=v_rf,
    frequency_rf=f_rf,
    lag_rf=0,
    length=100.)

line = xt.Line(elements=[otm], particle_ref=particle_ref)

tw = line.twiss()

delta_test = np.linspace(0, 5e-3, 20)
p = line.build_particles(delta=delta_test)

line.track(p, turn_by_turn_monitor=True, num_turns=5000)
mon = line.record_last_track

plt.close('all')
plt.figure(1)
plt.plot(mon.zeta.T, mon.delta.T, color='C0')
plt.xlabel('zeta [m]')
plt.ylabel('delta')

plt.show()