import xtrack as xt
import xdeps as xd
import numpy as np

env = xt.load_madx_lattice('../../test_data/sps_thick/sps.seq')
env.vars.load_madx('../../test_data/sps_thick/lhc_q20.str')

line = env.sps
line.particle_ref = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1, p0c=20e9,
                                 anomalous_magnetic_moment=0.00115965218128)

line['qf.62410'].shift_y = 1e-3

tt = line.get_table()
tt_quad = tt.rows[tt.element_type == 'Quadrupole']

line.slice_thick_elements(
    slicing_strategies=[
        xt.Strategy(None),
        xt.Strategy(xt.Teapot(5, mode='thick'), element_type=xt.Quadrupole),
    ])

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

def spin_fixed_point(s):

    pp = tw.particle_on_co.copy()

    thata = s[0]
    phi = s[1]

    pp.spin_x = np.cos(thata) * np.cos(phi)
    pp.spin_y = np.sin(thata) * np.cos(phi)
    pp.spin_z = np.sin(phi)
    # pp.get_table().cols['spin_x spin_y spin_z'].show()

    pp0 = pp.copy()
    # pp0.get_table().cols['spin_x spin_y spin_z'].show()

    line.track(pp)
    # pp.get_table().cols['spin_x spin_y spin_z'].show()
    # print('spin', pp.spin_x[0], pp.spin_y[0], pp.spin_z[0])
    # print('spin0', pp0.spin_x[0], pp0.spin_y[0], pp0.spin_z[0])

    return np.array([pp.spin_x[0] - pp0.spin_x[0],
                     pp.spin_y[0] - pp0.spin_y[0],
                     pp.spin_z[0] - pp0.spin_z[0]])


opt = xd.Optimize.from_callable(spin_fixed_point, x0=(0., 0.),
                                steps=[1e-4, 1e-4],
                                tar=[0., 0.],
                                limits=[[-2*np.pi, 2*np.pi],
                                        [-np.pi/2, np.pi/2]],
                                tols=[1e-12, 1e-12])
opt.run_nelder_mead()
opt.step(10)
opt.run_nelder_mead()
opt.solve()

theta = opt.get_knob_values()[0]
phi = opt.get_knob_values()[1]
p_n0b = tw.particle_on_co.copy()
p_n0b.spin_x = np.cos(theta) * np.cos(phi)
p_n0b.spin_y = np.sin(theta) * np.cos(phi)
p_n0b.spin_z = np.sin(phi)

line.track(p_n0b, turn_by_turn_monitor='ONE_TURN_EBE')
mon0b = line.record_last_track

prrrr


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
plt.plot(mon0.s[0, :], mon0b.spin_x[0, :], '.-', label='spin_x')
plt.plot(mon0.s[0, :], mon0b.spin_y[0, :], '.-', label='spin_y')
plt.plot(mon0.s[0, :], mon0b.spin_z[0, :], '.-', label='spin_z')
plt.xlabel('s [m]')
plt.ylabel('spin component')
plt.legend()


plt.show()