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

line.freeze_longitudinal()

line.vars['i_oct_b1'] = 500

plt.close('all')
plt.figure(1)

# Compute and plot footprint
fp1 = line.get_footprint(nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                         freeze_longitudinal=True)
fp1.plot(color='r', label='I_oct=500')

plt.legend()

plt.show()