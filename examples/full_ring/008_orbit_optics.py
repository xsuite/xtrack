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

for ee in line.elements:
    if ee.__class__.__name__.startswith('BeamBeam'):
        assert hasattr(ee, 'q0')
        ee.q0 = 0

print('Build tracker...')
freeze_vars = xp.particles.part_energy_varnames() + ['zeta']
tracker = xt.Tracker(_context=context,
            line=line,
            local_particle_src=xp.gen_local_particle_api(
                                                freeze_vars=freeze_vars),
            )


part0 = xp.Particles(_context=context, **input_data['particle'])

part_on_co = tracker.find_closed_orbit(part0)

dx=1e-7; dpx=1e-10; dy=1e-7; dpy=1e-10; dzeta=1e-6; ddelta=1e-7

num_elements = len(tracker.line.elements)
x = np.zeros(num_elements, dtype=np.float64)
betx = np.zeros(num_elements, dtype=np.float64)
bety = np.zeros(num_elements, dtype=np.float64)
s = tracker.line.get_s_elements()

for ii, ee in enumerate(tracker.line.elements):

    print(f'{ii}/{len(tracker.line.elements)}        ', end='\r', flush=True)

    part_temp = xp.build_particles(particle_ref=part_on_co, mode='shift',
            x  =    [dx,  0., 0.,  0.,    0.,     0., -dx,   0.,  0.,   0.,     0.,      0.],
            px =    [0., dpx, 0.,  0.,    0.,     0.,  0., -dpx,  0.,   0.,     0.,      0.],
            y  =    [0.,  0., dy,  0.,    0.,     0.,  0.,   0., -dy,   0.,     0.,      0.],
            py =    [0.,  0., 0., dpy,    0.,     0.,  0.,   0.,  0., -dpy,     0.,      0.],
            zeta =  [0.,  0., 0.,  0., dzeta,     0.,  0.,   0.,  0.,   0., -dzeta,      0.],
            delta = [0.,  0., 0.,  0.,    0., ddelta,  0.,   0.,  0.,   0.,     0., -ddelta],)

    tracker.track(part_temp, ele_start=ii)
    tracker.track(part_temp, ele_start=0, num_elements=ii)

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

    WW, WWinv, Rot = xp.compute_linear_normal_form(RR, tol_det_M=10e-2)

    x[ii] = part_on_co.x[0]
    betx[ii] = WW[0, 0]**2 + WW[0, 1]**2
    bety[ii] = WW[2, 2]**2 + WW[2, 3]**2

    ee.track(part_on_co)


