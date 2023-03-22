import numpy as np

import xtrack as xt
import xpart as xp

import matplotlib.pyplot as plt

nemitt_x = 1e-6
nemitt_y = 1e-6


line = xt.Line.from_json(
    '../../test_data/hllhc15_noerrors_nobb/line_w_knobs_and_particle.json')
line.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, p0c=7e12)
line.build_tracker()

plt.close('all')
plt.figure(1)

fp0 = line.get_footprint(nemitt_x=nemitt_x, nemitt_y=nemitt_y)
fp0.plot(color='k', label='I_oct=0')

line.vars['i_oct_b1'] = 500
fp1 = line.get_footprint(nemitt_x=nemitt_x, nemitt_y=nemitt_y)
fp1.plot(color='r', label='I_oct=500')

line.vars['i_oct_b1'] = -250
fp2 = line.get_footprint(nemitt_x=nemitt_x, nemitt_y=nemitt_y)
fp2.plot(color='b', label='I_oct=-250')

plt.legend()

plt.figure(2)

line.vars['i_oct_b1'] = 0
fp0 = line.get_footprint(nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                         mode='uniform_action_grid')
fp0.plot(color='k', label='I_oct=0')

line.vars['i_oct_b1'] = 500
fp1 = line.get_footprint(nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                            mode='uniform_action_grid')
fp1.plot(color='r', label='I_oct=500')

line.vars['i_oct_b1'] = -250
fp2 = line.get_footprint(nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                         mode='uniform_action_grid')
fp2.plot(color='b', label='I_oct=-250')

plt.legend()

plt.show()