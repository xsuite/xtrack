import numpy as np

import xtrack as xt
import xpart as xp

line = xt.Line.from_json(
    '../../test_data/hllhc14_no_errors_with_coupling_knobs/line_b1.json')
line.build_tracker()

x_normalized, y_normalized, r_xy, theta_xy = xp.generate_2D_polar_grid(
    r_range=(0, 50.), # beam sigmas
    theta_range=(0, np.pi/2), nr=50, ntheta=50)

delta_init = 0 # In case off-momentum DA is needed

particles = line.build_particles(x_norm=x_normalized, px_norm=0,
                                 y_norm=y_normalized, py_norm=0,
                                 nemitt_x=2e-6, nemitt_y=2e-6,
                                 delta=delta_init)

line.track(particles, num_turns=100, time=True)

particles.sort(interleave_lost_particles=True)

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.scatter(x_normalized, y_normalized, c=particles.at_turn)
