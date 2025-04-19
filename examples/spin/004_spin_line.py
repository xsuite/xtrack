import xtrack as xt
import xdeps as xd
import numpy as np

# env = xt.load_madx_lattice('../../test_data/sps_thick/sps.seq')
# env.vars.load_madx('../../test_data/sps_thick/lhc_q20.str')
# line = env.sps
# line.particle_ref = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1, p0c=20e9,
#                                  )
# line['qf.62410'].shift_y = 1e-3

line = xt.Line.from_json('lep_corrected_sol.json')
line.particle_ref.anomalous_magnetic_moment=0.00115965218128
spin_tune = line.particle_ref.anomalous_magnetic_moment[0]*line.particle_ref.gamma0[0]

line['sol_l_ip2'].ks *= -1
line['sol_r_ip2'].ks *= -1
line['sol_l_ip4'].ks *= -1
line['sol_r_ip4'].ks *= -1
line['sol_l_ip6'].ks *= -1
line['sol_r_ip6'].ks *= -1
line['sol_l_ip8'].ks *= -1
line['sol_r_ip8'].ks *= -1

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

def spin_fixed_point(s):

    pp = tw.particle_on_co.copy()

    pp.spin_x = s[0]
    pp.spin_z = s[1]
    pp.spin_y = np.sqrt(1 - s[0]**2 - s[1]**2)
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
                                limits=[(-1, 1), (-1, 1)],
                                tols=[1e-12, 1e-12],
                                show_call_counter=False)
opt.solve(verbose=False)

tw_spin = line.twiss(
    spin=True,
    betx=1., bety=1.,
    x=tw.x[0], px=tw.px[0], y=tw.y[0], py=tw.py[0],
    zeta=tw.zeta[0], delta=tw.delta[0],
    spin_x=opt.get_knob_values()[0],
    spin_z=opt.get_knob_values()[1],
    spin_y=np.sqrt(1 - opt.get_knob_values()[0]**2 - opt.get_knob_values()[1]**2))

p_n0 = tw.particle_on_co.copy()
p_n0.spin_x = opt.get_knob_values()[0]
p_n0.spin_z = opt.get_knob_values()[1]
p_n0.spin_y = np.sqrt(1 - p_n0.spin_x**2 - p_n0.spin_z**2)

line.track(p_n0, turn_by_turn_monitor='ONE_TURN_EBE')
mon0 = line.record_last_track

mask = (mon0.s[0, :] > 9997) & (mon0.s[0, :] < 11200)




import matplotlib.pyplot as plt
plt.close('all')
fig1 = plt.figure(1)
tw.plot(lattice_only=True, figure=fig1)
plt.plot(mon.s[0, :], mon.spin_x[0, :], label='spin_x')
plt.plot(mon.s[0, :], mon.spin_z[0, :], label='spin_z')
plt.xlabel('s [m]')
plt.ylabel('spin component')
plt.legend()

fig2 = plt.figure(2)
tw.plot(lattice_only=True, figure=fig2)
plt.plot(mon0.s[0, :], mon0.spin_x[0, :], '.-', label='spin_x')
# plt.plot(mon0.s[0, :], mon0.spin_y[0, :], '.-', label='spin_y')
plt.plot(mon0.s[0, :], mon0.spin_z[0, :], '.-', label='spin_z')
plt.xlabel('s [m]')
plt.ylabel('spin component')
plt.legend()

plt.figure(3)
plt.plot(-mon0.spin_z[0, mask], -mon0.spin_x[0, mask], 'x-', label='spin_x')
plt.xlabel(r'$n_{0z}$')
plt.ylabel(r'$n_{0x}$')
plt.axis('equal')
plt.suptitle('IP4 right')

plt.show()