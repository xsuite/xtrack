# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import json
import numpy as np

import xobjects as xo
import xtrack as xt
import xpart as xp

#####################################
# Load a line and build the tracker #
#####################################

fname_line_particles = '../../test_data/hllhc15_noerrors_nobb/line_and_particle.json'
#fname_line_particles = '../../test_data/sps_w_spacecharge/line_no_spacecharge_and_particle.json' #!skip-doc

with open(fname_line_particles, 'r') as fid:
    input_data = json.load(fid)
line = xt.Line.from_dict(input_data['line'])
line.particle_ref = xp.Particles.from_dict(input_data['particle'])

tracker = line.build_tracker()

particles = tracker.build_particles(
    nemitt_x=2.5e-6, nemitt_y=1e-6,
    x_norm=[-1, 0, 0.5], y_norm=[0.3, -0.2, 0.2],
    px_norm=[0.1, 0.2, 0.3], py_norm=[0.5, 0.6, 0.8],
    zeta=[0, 0.1, -0.1], delta=[1e-4, 0., -1e-4])

tw = tracker.twiss()

norm_coord = tw.get_normalized_coordinates(particles, nemitt_x=2.5e-6,
                                           nemitt_y=1e-6)

assert np.allclose(norm_coord['x_norm'], [-1, 0, 0.5], atol=1e-14, rtol=0)
assert np.allclose(norm_coord['y_norm'], [0.3, -0.2, 0.2], atol=1e-14, rtol=0)
assert np.allclose(norm_coord['px_norm'], [0.1, 0.2, 0.3], atol=1e-14, rtol=0)
assert np.allclose(norm_coord['py_norm'], [0.5, 0.6, 0.8], atol=1e-14, rtol=0)


# Introduce a non-zero closed orbit
line['mqwa.a4r3.b1..1'].knl[0] = 10e-6
line['mqwa.a4r3.b1..1'].ksl[0] = 5e-6

particles1 = tracker.build_particles(
    nemitt_x=2.5e-6, nemitt_y=1e-6,
    x_norm=[-1, 0, 0.5], y_norm=[0.3, -0.2, 0.2],
    px_norm=[0.1, 0.2, 0.3], py_norm=[0.5, 0.6, 0.8],
    zeta=[0, 0.1, -0.1], delta=[1e-4, 0., -1e-4])

tw1 = tracker.twiss()
norm_coord1 = tw1.get_normalized_coordinates(particles1, nemitt_x=2.5e-6,
                                             nemitt_y=1e-6)

assert np.allclose(norm_coord1['x_norm'], [-1, 0, 0.5], atol=1e-14, rtol=0)
assert np.allclose(norm_coord1['y_norm'], [0.3, -0.2, 0.2], atol=1e-14, rtol=0)
assert np.allclose(norm_coord1['px_norm'], [0.1, 0.2, 0.3], atol=1e-14, rtol=0)
assert np.allclose(norm_coord1['py_norm'], [0.5, 0.6, 0.8], atol=1e-14, rtol=0)