# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np

import xobjects as xo
import xtrack as xt

context = xo.ContextCpu()

n_part=300000

particles = xt.Particles(_context=context,
        p0c=6500e9,
        x=np.random.uniform(-0.25, 0.25, n_part),
        px = np.zeros(n_part),
        y=np.random.uniform(0, 0.1, n_part),
        py = np.zeros(n_part))

tilt_deg = 10.
aper = xt.LimitRect(_context=context,
                    min_x=-.1,
                    max_x=.1,
                    min_y=-0.001,
                    max_y=0.001,
                    shift_x=0.08,
                    shift_y=0.04,
                    rot_s_rad=np.deg2rad(tilt_deg))

aper.track(particles)

part_id = context.nparray_from_context_array(particles.particle_id)
part_state = context.nparray_from_context_array(particles.state)
part_x = context.nparray_from_context_array(particles.x)
part_y = context.nparray_from_context_array(particles.y)

x_alive = part_x[part_state>0]
y_alive = part_y[part_state>0]

assert_allclose = np.testing.assert_allclose
assert_allclose(np.mean(x_alive), 0.08, rtol=5e-2, atol=0)
assert_allclose(np.mean(y_alive), 0.04, rtol=5e-2, atol=0)
slope = np.polyfit(x_alive, y_alive, 1)[0]
assert_allclose(slope, np.tan(np.deg2rad(tilt_deg)), rtol=5e-2, atol=0)

# Correlation coefficient


id_alive = part_id[part_state>0]

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.plot(part_x, part_y, '.', color='red')
plt.plot(part_x[part_state>0], part_y[part_state>0], '.', color='green')
plt.axis('equal')

plt.show()
