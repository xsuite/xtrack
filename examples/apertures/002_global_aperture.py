import numpy as np

import xobjects as xo
import xtrack as xt

import pysixtrack

context = xo.ContextCpu()
context = xo.ContextCupy()
context = xo.ContextPyopencl()

x_aper_min = -0.1
x_aper_max = 0.2
y_aper_min = 0.2
y_aper_max = 0.3

part_gen_range = 0.35
n_part=100

pyst_part = pysixtrack.Particles(
        p0c=6500e9,
        x=np.zeros(n_part),
        px=np.linspace(-1, 1, n_part),
        y=np.zeros(n_part),
        py=np.linspace(-2, 2, n_part),
        sigma=np.zeros(n_part),
        delta=np.zeros(n_part))
pyst_part.px[1::2] = 0
pyst_part.py[0::2] = 0

particles = xt.Particles(_context=context, pysixtrack_particles=pyst_part)

# Build a small test line
tot_length = 2.
n_slices = 10000
pyst_line = pysixtrack.Line(elements=
                n_slices*[pysixtrack.elements.Drift(length=tot_length/n_slices)],
                element_names=['drift{ii}' for ii in range(n_slices)])

tracker = xt.Tracker(context=context, sequence=pyst_line)

tracker.track(particles)

part_id = context.nparray_from_context_array(particles.particle_id)
part_state = context.nparray_from_context_array(particles.state)
part_x = context.nparray_from_context_array(particles.x)
part_y = context.nparray_from_context_array(particles.y)
part_px = context.nparray_from_context_array(particles.px)
part_py = context.nparray_from_context_array(particles.py)
part_s = context.nparray_from_context_array(particles.s)

id_alive = part_id[part_state>0]

#x = px*s
s_expected = []
for ii in range(n_part):
    if np.abs(part_px[ii]) * tot_length > tracker.global_xy_limit:
        s_expected_x = np.abs(tracker.global_xy_limit / part_px[ii])
    else:
        s_expected_x = tot_length

    if np.abs(part_py[ii] * tot_length) > tracker.global_xy_limit:
        s_expected_y = np.abs(tracker.global_xy_limit / part_py[ii])
    else:
        s_expected_y = tot_length

    if s_expected_x<s_expected_y:
        s_expected.append(s_expected_x)
    else:
        s_expected.append(s_expected_y)

s_expected = np.array(s_expected)

assert np.allclose(part_s, s_expected, atol=1e-3)
