import xtrack as xt
import xdeps as xd
import numpy as np
import xobjects as xo

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

tw = line.twiss4d(spin=True)
tw_ir4 = tw.rows[9997:11200:'s']

xo.assert_allclose(tw.spin_x**2 + tw.spin_y**2 + tw.spin_z**2,
                   1, atol=1e-12, rtol=0)
xo.assert_allclose(tw.spin_x[0], tw.spin_x[-1], atol=1e-10, rtol=0)
xo.assert_allclose(tw.spin_y[0], tw.spin_y[-1], atol=1e-10, rtol=0)
xo.assert_allclose(tw.spin_z[0], tw.spin_z[-1], atol=1e-10, rtol=0)

import matplotlib.pyplot as plt
plt.close('all')
fig2 = plt.figure(2)
tw.plot(lattice_only=True, figure=fig2)
plt.plot(tw.s, tw.spin_x, '-', label='spin_x')
plt.plot(tw.s, tw.spin_z, '-', label='spin_y')
plt.xlabel('s [m]')
plt.ylabel('spin component')
plt.legend()

plt.figure(3)
plt.plot(-tw_ir4.spin_z, -tw_ir4.spin_x, 'x-', label='spin')
plt.xlabel(r'$n_{0z}$')
plt.ylabel(r'$n_{0x}$')
plt.axis('equal')
plt.suptitle('IP4 right')

plt.show()