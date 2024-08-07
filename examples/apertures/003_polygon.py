# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np

import xobjects as xo
import xtrack as xt

context = xo.ContextCpu()

np2ctx = context.nparray_to_context_array
ctx2np = context.nparray_from_context_array


x_vertices=np.array([1.5, 0.2, -1, -1,  1])*1e-2
y_vertices=np.array([1.3, 0.5,  1, -1, -1])*1e-2

aper = xt.LimitPolygon(
                _context=context,
                x_vertices=np2ctx(x_vertices),
                y_vertices=np2ctx(y_vertices))

# Try some particles inside
parttest = xt.Particles(
                _context=context,
                p0c=6500e9,
                x=x_vertices*0.99,
                y=y_vertices*0.99)
aper.track(parttest)
assert np.allclose(ctx2np(parttest.state), 1)

# Try some particles outside
parttest = xt.Particles(
                _context=context,
                p0c=6500e9,
                x=x_vertices*1.01,
                y=y_vertices*1.01)
aper.track(parttest)
assert np.allclose(ctx2np(parttest.state), 0)

part_gen_range = 0.02
n_part=10000
particles = xt.Particles(
                _context=context,
                p0c=6500e9,
                x=np.random.uniform(-part_gen_range, part_gen_range, n_part),
                px = np.zeros(n_part),
                y=np.random.uniform(-part_gen_range, part_gen_range, n_part),
                py = np.zeros(n_part),
                zeta = np.zeros(n_part),
                delta = np.zeros(n_part))

aper.track(particles)

part_id = context.nparray_from_context_array(particles.particle_id)
part_state = context.nparray_from_context_array(particles.state)
part_x = context.nparray_from_context_array(particles.x)
part_y = context.nparray_from_context_array(particles.y)

import matplotlib.pyplot as plt
plt.close('all')
fig1 = plt.figure(1)
ax = fig1.add_subplot(111)
plt.plot(part_x, part_y, '.', color='red')
plt.plot(part_x[part_state>0], part_y[part_state>0], '.', color='green')
ax.plot(aper.x_closed, aper.y_closed, linewidth=3, color='black')

plt.show()
