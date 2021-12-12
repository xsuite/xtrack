import pickle
import json
import pathlib
import numpy as np

import xobjects as xo
import xtrack as xt
import xpart as xp

from make_short_line import make_short_line

short_test = False # Short line (5 elements)

test_data_folder = pathlib.Path(
        __file__).parent.joinpath('../../test_data').absolute()

fname_line_particles = test_data_folder.joinpath(
                        './hllhc_14/line_and_particle.json')

####################
# Choose a context #
####################

context = xo.ContextCpu()


with open(fname_line_particles, 'r') as fid:
    input_data = json.load(fid)

line = xt.Line.from_dict(input_data['line'])
if short_test:
    line = make_short_line(line)

print('Build tracker...')
freeze_vars = xp.particles.part_energy_varnames() + ['zeta']
tracker = xt.Tracker(_context=context,
            line=line,
            local_particle_src=xp.gen_local_particle_api(
                                                freeze_vars=freeze_vars),
            )

self = tracker

part0 = xp.Particles(_context=context, **input_data['particle'])

part_on_co = tracker.find_closed_orbit(part0)

RR_ref =  tracker.compute_one_turn_matrix_finite_differences(part_on_co)

dx=1e-7; dpx=1e-10; dy=1e-7; dpy=1e-10; dzeta=1e-6; ddelta=1e-7

def test():
    particle_on_co = part_on_co.copy(
                        _context=self._buffer.context)

    part_temp = xp.build_particles(particle_ref=part_on_co, mode='shift',
            x  =    [dx,  0., 0.,  0.,    0.,     0., -dx,   0.,  0.,   0.,     0.,      0.],
            px =    [0., dpx, 0.,  0.,    0.,     0.,  0., -dpx,  0.,   0.,     0.,      0.],
            y  =    [0.,  0., dy,  0.,    0.,     0.,  0.,   0., -dy,   0.,     0.,      0.],
            py =    [0.,  0., 0., dpy,    0.,     0.,  0.,   0.,  0., -dpy,     0.,      0.],
            zeta =  [0.,  0., 0.,  0., dzeta,     0.,  0.,   0.,  0.,   0., -dzeta,      0.],
            delta = [0.,  0., 0.,  0.,    0., ddelta,  0.,   0.,  0.,   0.,     0., -ddelta],)

    tracker.track(part_temp)

    temp_mat = np.zeros(shape=(6, 12), dtype=np.float64)
    temp_mat[0, :] = part_temp.x
    temp_mat[1, :] = part_temp.px
    temp_mat[2, :] = part_temp.y
    temp_mat[3, :] = part_temp.py
    temp_mat[4, :] = part_temp.zeta
    temp_mat[5, :] = part_temp.delta

    RR = np.zeros(shape=(6, 6), dtype=np.float64)

    for jj, dd in enumerate([dx, dpx, dy, dpy, dzeta, ddelta]):
        RR[:, jj] = (temp_mat[:, jj] - temp_mat[:, jj+6])/(2*dd)

    return RR

