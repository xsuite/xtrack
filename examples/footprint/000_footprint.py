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

# Compute and plot footprint
fp0 = line.get_footprint(nemitt_x=nemitt_x, nemitt_y=nemitt_y)
fp0.plot(color='k', label='I_oct=0')

# Change octupoles strength and compute footprint again
line.vars['i_oct_b1'] = 500
fp1 = line.get_footprint(nemitt_x=nemitt_x, nemitt_y=nemitt_y)
fp1.plot(color='r', label='I_oct=500')

# Change octupoles strength and compute footprint again
line.vars['i_oct_b1'] = -250
fp2 = line.get_footprint(nemitt_x=nemitt_x, nemitt_y=nemitt_y)
fp2.plot(color='b', label='I_oct=-250')

plt.legend()

plt.figure(2)

line.vars['i_oct_b1'] = 0
fp0_jgrid = line.get_footprint(nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                         mode='uniform_action_grid')
fp0_jgrid.plot(color='k', label='I_oct=0')

# Change octupoles strength and compute footprint again
line.vars['i_oct_b1'] = 500
fp1_jgrid = line.get_footprint(nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                            mode='uniform_action_grid')
fp1_jgrid.plot(color='r', label='I_oct=500')

# Change octupoles strength and compute footprint again
line.vars['i_oct_b1'] = -250
fp2_jgrid = line.get_footprint(nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                         mode='uniform_action_grid')
fp2_jgrid.plot(color='b', label='I_oct=-250')

plt.legend()
plt.show()

#!end-doc-part

assert hasattr(fp0, 'theta_grid')
assert hasattr(fp0, 'r_grid')

assert np.isclose(fp0.r_grid[0], 0.1, rtol=0, atol=1e-10)
assert np.isclose(fp0.r_grid[-1], 6, rtol=0, atol=1e-10)

assert np.isclose(fp0.theta_grid[0], 0.05, rtol=0, atol=1e-10)
assert np.isclose(fp0.theta_grid[-1], np.pi/2-0.05, rtol=0, atol=1e-10)


