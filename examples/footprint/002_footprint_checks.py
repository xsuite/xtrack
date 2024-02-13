import numpy as np

import xtrack as xt

import matplotlib.pyplot as plt

nemitt_x = 1e-6
nemitt_y = 1e-6


line = xt.Line.from_json(
    '../../test_data/hllhc15_noerrors_nobb/line_w_knobs_and_particle.json')
line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, p0c=7e12)
line.build_tracker()

plt.close('all')
plt.figure(1)

line.vars['i_oct_b1'] = 0
fp0 = line.get_footprint(nemitt_x=nemitt_x, nemitt_y=nemitt_y)
fp0.plot(color='k', label='I_oct=0')

line.vars['i_oct_b1'] = 500
fp1 = line.get_footprint(nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                         n_r=11, n_theta=7, r_range=[0.05, 7],
                         theta_range=[0.01, np.pi/2-0.01])
fp1.plot(color='r', label='I_oct=500')

plt.legend()


assert hasattr(fp1, 'theta_grid')
assert hasattr(fp1, 'r_grid')

assert len(fp1.r_grid) == 11
assert len(fp1.theta_grid) == 7

assert np.isclose(fp1.r_grid[0], 0.05, rtol=0, atol=1e-10)
assert np.isclose(fp1.r_grid[-1], 7, rtol=0, atol=1e-10)

assert np.isclose(fp1.theta_grid[0], 0.01, rtol=0, atol=1e-10)
assert np.isclose(fp1.theta_grid[-1], np.pi/2 - 0.01, rtol=0, atol=1e-10)

#i_theta = 0, i_r = 0
assert np.isclose(fp1.x_norm_2d[0, 0], 0.05, rtol=0, atol=1e-3)
assert np.isclose(fp1.y_norm_2d[0, 0], 0, rtol=0, atol=1e-3)
assert np.isclose(fp1.qx[0, 0], 0.31, rtol=0, atol=5e-5)
assert np.isclose(fp1.qy[0, 0], 0.32, rtol=0, atol=5e-5)

#i_theta = 0, i_r = 10
assert np.isclose(fp1.x_norm_2d[0, -1], 7, rtol=0, atol=1e-3)
assert np.isclose(fp1.y_norm_2d[0, -1], 0.07, rtol=0, atol=1e-3)
assert np.isclose(fp1.qx[0, -1], 0.3129, rtol=0, atol=2e-4)
assert np.isclose(fp1.qy[0, -1], 0.3185, rtol=0, atol=2e-4)

#i_theta = 6, i_r = 0
assert np.isclose(fp1.x_norm_2d[-1, 0], 0, rtol=0, atol=1e-3)
assert np.isclose(fp1.y_norm_2d[-1, 0], 0.05, rtol=0, atol=1e-3)
assert np.isclose(fp1.qx[0, 0], 0.31, rtol=0, atol=5e-5)
assert np.isclose(fp1.qy[0, 0], 0.32, rtol=0, atol=5e-5)

#i_theta = 6, i_r = 10
assert np.isclose(fp1.x_norm_2d[-1, -1], 0.07, rtol=0, atol=1e-3)
assert np.isclose(fp1.y_norm_2d[-1, -1], 7, rtol=0, atol=1e-3)
assert np.isclose(fp1.qx[-1, -1], 0.3085, rtol=0, atol=2e-4)
assert np.isclose(fp1.qy[-1, -1], 0.3229, rtol=0, atol=2e-4)

assert np.isclose(np.max(fp1.qx[:]) - np.min(fp1.qx[:]), 4.4e-3, rtol=0, atol=1e-4)
assert np.isclose(np.max(fp1.qy[:]) - np.min(fp1.qy[:]), 4.4e-3, rtol=0, atol=1e-4)


line.vars['i_oct_b1'] = 0
fp0_jgrid = line.get_footprint(nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                         mode='uniform_action_grid')
fp0_jgrid.plot(color='k', label='I_oct=0')

assert hasattr(fp0, 'theta_grid')
assert hasattr(fp0, 'r_grid')

assert np.isclose(fp0.r_grid[0], 0.1, rtol=0, atol=1e-10)
assert np.isclose(fp0.r_grid[-1], 6, rtol=0, atol=1e-10)

assert np.isclose(fp0.theta_grid[0], 0.05, rtol=0, atol=1e-10)
assert np.isclose(fp0.theta_grid[-1], np.pi/2-0.05, rtol=0, atol=1e-10)

assert len(fp0.r_grid) == 10
assert len(fp0.theta_grid) == 10

#i_theta = 0, i_r = 0
assert np.isclose(fp0.x_norm_2d[0, 0], 0.1, rtol=0, atol=1e-3)
assert np.isclose(fp0.y_norm_2d[0, 0], 0.005, rtol=0, atol=1e-3)
assert np.isclose(fp0.qx[0, 0], 0.31, rtol=0, atol=5e-5)
assert np.isclose(fp0.qy[0, 0], 0.32, rtol=0, atol=5e-5)

assert np.isclose(np.max(fp0.qx[:]) - np.min(fp0.qx[:]), 0.0003, rtol=0, atol=2e-5)
assert np.isclose(np.max(fp0.qy[:]) - np.min(fp0.qy[:]), 0.0003, rtol=0, atol=2e-5)

plt.figure(2)

line.vars['i_oct_b1'] = 500
fp1_jgrid = line.get_footprint(nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                            x_norm_range=[0.01, 6], y_norm_range=[0.01, 6],
                            n_x_norm=9, n_y_norm=8,
                            mode='uniform_action_grid')
fp1_jgrid.plot(color='r', label='I_oct=500')

assert hasattr(fp1_jgrid,  'Jx_grid')
assert hasattr(fp1_jgrid,  'Jy_grid')

assert len(fp1_jgrid.Jx_grid) == 9
assert len(fp1_jgrid.Jy_grid) == 8

assert np.allclose(np.diff(fp1_jgrid.Jx_grid), np.diff(fp1_jgrid.Jx_grid)[0],
                   rtol=0, atol=1e-10)
assert np.allclose(np.diff(fp1_jgrid.Jy_grid), np.diff(fp1_jgrid.Jy_grid)[0],
                     rtol=0, atol=1e-10)

assert np.isclose(fp1_jgrid.x_norm_2d[0, 0], 0.01, rtol=0, atol=1e-10)
assert np.isclose(fp1_jgrid.y_norm_2d[0, 0], 0.01, rtol=0, atol=1e-10)
assert np.isclose(fp1_jgrid.qx[0, 0], 0.31, rtol=0, atol=5e-5)
assert np.isclose(fp1_jgrid.qy[0, 0], 0.32, rtol=0, atol=5e-5)

assert np.isclose(fp1_jgrid.x_norm_2d[0, -1], 6, rtol=0, atol=1e-10)
assert np.isclose(fp1_jgrid.y_norm_2d[0, -1], 0.01, rtol=0, atol=1e-10)
assert np.isclose(fp1_jgrid.qx[0, -1], 0.3121, rtol=0, atol=1e-4)
assert np.isclose(fp1_jgrid.qy[0, -1], 0.3189, rtol=0, atol=1e-4)

assert np.isclose(fp1_jgrid.x_norm_2d[-1, 0], 0.01, rtol=0, atol=1e-10)
assert np.isclose(fp1_jgrid.y_norm_2d[-1, 0], 6, rtol=0, atol=1e-10)
assert np.isclose(fp1_jgrid.qx[-1, 0], 0.3089, rtol=0, atol=1e-4)
assert np.isclose(fp1_jgrid.qy[-1, 0], 0.3221, rtol=0, atol=1e-4)

assert np.isclose(fp1_jgrid.x_norm_2d[-1, -1], 6, rtol=0, atol=1e-10)
assert np.isclose(fp1_jgrid.y_norm_2d[-1, -1], 6, rtol=0, atol=1e-10)
assert np.isclose(fp1_jgrid.qx[-1, -1], 0.3111, rtol=0, atol=1e-4)
assert np.isclose(fp1_jgrid.qy[-1, -1], 0.3210, rtol=0, atol=1e-4)

assert np.isclose(np.max(fp1_jgrid.qx[:]) - np.min(fp1_jgrid.qx[:]), 0.0032,
                    rtol=0, atol=1e-4)
assert np.isclose(np.max(fp1_jgrid.qy[:]) - np.min(fp1_jgrid.qy[:]), 0.0032,
                    rtol=0, atol=1e-4)

x_norm_range = [1, 6]
y_norm_range = [1, 6]
line.vars['i_oct_b1'] = 50000 # Particles are lost for such high octupole current
fp50k = line.get_footprint(nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                            x_norm_range=x_norm_range, y_norm_range=y_norm_range,
                            n_x_norm=9, n_y_norm=8,
                            mode='uniform_action_grid',
                            linear_rescale_on_knobs=xt.LinearRescale(
                                knob_name='i_oct_b1', v0=500, dv=100))


line.vars['i_oct_b1'] = 60000 # Particles are lost for such high octupole current
fp60k = line.get_footprint(nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                            x_norm_range=x_norm_range, y_norm_range=y_norm_range,
                            n_x_norm=9, n_y_norm=8,
                            mode='uniform_action_grid',
                            linear_rescale_on_knobs=xt.LinearRescale(
                            knob_name='i_oct_b1', v0=500, dv=100))

line.vars['i_oct_b1'] = 500
fp500 = line.get_footprint(nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                            x_norm_range=x_norm_range, y_norm_range=y_norm_range,
                            n_x_norm=9, n_y_norm=8,
                            mode='uniform_action_grid')

line.vars['i_oct_b1'] = 600
fp600 = line.get_footprint(nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                            x_norm_range=x_norm_range, y_norm_range=y_norm_range,
                            n_x_norm=9, n_y_norm=8,
                            mode='uniform_action_grid')

assert np.allclose((fp60k.qx - fp50k.qx)/(fp600.qx-fp500.qx), 100, rtol=0, atol=1e-2)
assert np.allclose((fp60k.qy - fp50k.qy)/(fp600.qy-fp500.qy), 100, rtol=0, atol=1e-2)










