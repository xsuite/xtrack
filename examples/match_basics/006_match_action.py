import numpy as np
import xtrack as xt
import NAFFlib as nl

# Load a line and build a tracker
line = xt.Line.from_json('../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')
line.build_tracker()

nemitt_x = 2.5e-6
nemitt_y = 2.5e-6

k_oct_arr = np.arange(-250, 250, 1e1)
alpha_xx_arr = np.empty_like(k_oct_arr)
alpha_yy_arr = np.empty_like(k_oct_arr)
alpha_xy_arr = np.empty_like(k_oct_arr)
alpha_yx_arr = np.empty_like(k_oct_arr)

k_oct = 1e3

line.vars['kof.a23b1'] = k_oct
line.vars['kof.a34b1'] = k_oct
line.vars['kof.a67b1'] = k_oct
line.vars['kof.a78b1'] = k_oct

line.vars['kod.a23b1'] = k_oct
line.vars['kod.a34b1'] = k_oct
line.vars['kod.a67b1'] = k_oct
line.vars['kod.a78b1'] = k_oct

a0_sigmas = .01
a1_sigmas = 0.3
a2_sigmas = 0.2
num_turns = 256

Jx_0 = a0_sigmas**2 * nemitt_x / 2
Jx_1 = a1_sigmas**2 * nemitt_x / 2
Jx_2 = a2_sigmas**2 * nemitt_x / 2
Jy_0 = a0_sigmas**2 * nemitt_y / 2
Jy_1 = a1_sigmas**2 * nemitt_y / 2
Jy_2 = a2_sigmas**2 * nemitt_y / 2

particles = line.build_particles(
                    method='4d',
                    zeta=0, delta=0,
                    x_norm=[a1_sigmas, a2_sigmas, a0_sigmas, a0_sigmas],
                    y_norm=[a0_sigmas, a0_sigmas, a1_sigmas, a2_sigmas],
                    nemitt_x=nemitt_x, nemitt_y=nemitt_y)

line.track(particles, num_turns=num_turns, time=True, turn_by_turn_monitor=True)
mon = line.record_last_track

assert np.all(particles.state > 0)

qx = np.zeros(4)
qy = np.zeros(4)

for ii in range(len(qx)):
    qx[ii] = nl.get_tune(mon.x[ii, :])
    qy[ii] = nl.get_tune(mon.y[ii, :])

alpha_xx = (qx[1] - qx[0]) / (Jx_2 - Jx_1)
alpha_yy = (qy[3] - qy[2]) / (Jy_2 - Jy_1)
alpha_xy = (qx[3] - qx[2]) / (Jy_2 - Jy_1)
alpha_yx = (qy[1] - qy[0]) / (Jx_2 - Jx_1)

print(f'alpha_xx = {alpha_xx}, alpha_xy = {alpha_xy}')
print(f'alpha_yx = {alpha_yx}, alpha_yy = {alpha_yy}')
