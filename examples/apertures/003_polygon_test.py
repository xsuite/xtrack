import numpy as np

import xobjects as xo
import xtrack as xt

context = xo.ContextPyopencl()

aper = xt.LimitPolygon(
                _context=context,
                x_vertices=list(np.array([1.5, 0.2, -1, -1,  1])*1e-2),
                y_vertices=list(np.array([1.3, 0.5,  1, -1, -1])*1e-2))

part_gen_range = 0.02
n_part=10000
particles = xt.Particles(
                _context=context,
                p0c=6500e9,
                x=np.random.uniform(-part_gen_range, part_gen_range, n_part),
                px = np.zeros(n_part),
                y=np.random.uniform(-part_gen_range, part_gen_range, n_part),
                py = np.zeros(n_part),
                sigma = np.zeros(n_part),
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
