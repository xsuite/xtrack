import xtrack as xt
import numpy as np

env = xt.load_madx_lattice('../../test_data/sps_thick/sps.seq')
env.vars.load_madx('../../test_data/sps_thick/lhc_q20.str')

line = env.sps
line.particle_ref = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1, p0c=20e9,
                                 anomalous_magnetic_moment=0.00115965218128)

line['qf.62410'].shift_y = 1e-3

tt = line.get_table()
tt_quad = tt.rows[tt.element_type == 'Quadrupole']

line.set(tt_quad, model='mat-kick-mat', integrator='uniform', num_multipole_kicks=5)

tw = line.twiss4d()

p = line.build_particles(x=[0,0,0],
                         spin_x=[1,0,0],
                         spin_y=[0,1,0],
                         spin_z=[0,0,1])

line.config.XTRACK_MULTIPOLE_NO_SYNRAD = False # To enable spin tracking

line.track(p, turn_by_turn_monitor='ONE_TURN_EBE')
mon = line.record_last_track
m = np.array([p.spin_x, p.spin_y, p.spin_z]).T

eivals, eivec = np.linalg.eig(m)

# Identify index of eival closed to 1
i_ei = np.argmin(np.abs(eivals - 1))
# Get the corresponding eigenvector
v_ei = eivec[:, i_ei].real

p_n0 = tw.particle_on_co.copy()
p_n0.spin_x = v_ei[0]
p_n0.spin_y = v_ei[1]
p_n0.spin_z = v_ei[2]

line.track(p_n0, turn_by_turn_monitor='ONE_TURN_EBE')
mon0 = line.record_last_track


import matplotlib.pyplot as plt
plt.close('all')
fig1 = plt.figure(1)
tw.plot(lattice_only=True)
plt.plot(mon.s[0, :], mon.spin_x[0, :], label='spin_x')
plt.plot(mon.s[0, :], mon.spin_z[0, :], label='spin_z')
plt.xlabel('s [m]')
plt.ylabel('spin component')
plt.legend()

fig2 = plt.figure(2)
tw.plot(lattice_only=True)
plt.plot(mon0.s[0, :], mon0.spin_x[0, :], label='spin_x')
plt.plot(mon0.s[0, :], mon0.spin_z[0, :], label='spin_z')
plt.xlabel('s [m]')
plt.ylabel('spin component')
plt.legend()


plt.show()