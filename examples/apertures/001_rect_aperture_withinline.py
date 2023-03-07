# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np

import xobjects as xo
import xtrack as xt
import xpart as xp

import ducktrack as dtk

context = xo.ContextCpu()

x_aper_min = -0.1
x_aper_max = 0.2
y_aper_min = 0.2
y_aper_max = 0.3

part_gen_range = 0.35
n_part=10000

dtk_particles = dtk.TestParticles(
        p0c=6500e9,
        x=np.random.uniform(-part_gen_range, part_gen_range, n_part),
        px = np.zeros(n_part),
        y=np.random.uniform(-part_gen_range, part_gen_range, n_part),
        py = np.zeros(n_part))

particles = xp.Particles.from_dict(_context=context, dct=dtk_particles.to_dict())

aper_test = dtk.LimitRect(min_x=x_aper_min,
                                          max_x=x_aper_max,
                                          min_y=y_aper_min,
                                          max_y=y_aper_max)

aper = xt.LimitRect(_context=context,
                    **aper_test.to_dict())

aper_test.track(dtk_particles)

# Build a small test line
line = xt.Line(elements=[
                xt.Drift(length=5.),
                aper,
                xt.Drift(length=5.)],
                element_names=['drift0', 'aper', 'drift1'])

line.build_tracker(_context=context)

line.track(particles)

part_id = context.nparray_from_context_array(particles.particle_id)
part_state = context.nparray_from_context_array(particles.state)
part_x = context.nparray_from_context_array(particles.x)
part_y = context.nparray_from_context_array(particles.y)
part_s = context.nparray_from_context_array(particles.s)

id_alive = part_id[part_state>0]

assert np.allclose(np.sort(dtk_particles.particle_id), np.sort(id_alive))
assert np.allclose(part_s[part_state>0], 0.)
assert np.allclose(part_s[part_state<1], 5.)

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.plot(part_x, part_y, '.', color='red')
plt.plot(part_x[part_state>0], part_y[part_state>0], '.', color='green')

plt.show()
