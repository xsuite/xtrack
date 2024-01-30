# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np

import xtrack as xt
import xobjects as xo

context = xo.ContextCpu()

part_gen_range = 0.11
n_part=100000

particles = xt.Particles(_context=context,
        p0c=6500e9,
        x=np.random.uniform(-part_gen_range, part_gen_range, n_part),
        y=np.random.uniform(-part_gen_range, part_gen_range, n_part))

aper = xt.LimitRacetrack(_context=context,
                         min_x=-5e-2, max_x=10e-2,
                         min_y=-2e-2, max_y=4e-2,
                         a=2e-2, b=1e-2)

aper.track(particles)

part_state = context.nparray_from_context_array(particles.state)
part_x = context.nparray_from_context_array(particles.x)
part_y = context.nparray_from_context_array(particles.y)

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.plot(part_x, part_y, '.', color='red')
plt.plot(part_x[part_state>0], part_y[part_state>0], '.', color='green')


xy_out = np.array([
    [-4.8e-2, 3.7e-2],
    [9.6e-2, 3.7e-2],
    [-4.5e-2, -1.8e-2],
    [9.8e-2, -1.8e-2],
    ])

xy_in = np.array([
    [-4.2e-2, 3.3e-2],
    [9.4e-2, 3.6e-2],
    [-3.8e-2, -1.8e-2],
    [9.2e-2, -1.8e-2],
    ])

xy_all = np.concatenate([xy_out, xy_in], axis=0)

particles = xt.Particles(_context=context,
        p0c=6500e9,
        x=xy_all[:, 0],
        y=xy_all[:, 1])

aper.track(particles)

part_state = context.nparray_from_context_array(particles.state)
part_x = context.nparray_from_context_array(particles.x)
part_y = context.nparray_from_context_array(particles.y)
part_id = context.nparray_from_context_array(particles.particle_id)

assert np.all(part_state[part_id<4] == 0)
assert np.all(part_state[part_id>=4] == 1)

plt.plot(part_x, part_y, '.', color='k')
plt.plot(part_x[part_state>0], part_y[part_state>0], 'x', color='k')



plt.show()
