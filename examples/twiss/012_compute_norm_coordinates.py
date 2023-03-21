# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import json
import numpy as np

import xobjects as xo
import xtrack as xt
import xpart as xp

fname_line_particles = '../../test_data/hllhc15_noerrors_nobb/line_and_particle.json'

with open(fname_line_particles, 'r') as fid:
    input_data = json.load(fid)
line = xt.Line.from_dict(input_data['line'])
line.particle_ref = xp.Particles.from_dict(input_data['particle'])

line.build_tracker()

# Make some particles with known normalized coordinates
particles = line.build_particles(
    nemitt_x=2.5e-6, nemitt_y=1e-6,
    x_norm=[-1, 0, 0.5], y_norm=[0.3, -0.2, 0.2],
    px_norm=[0.1, 0.2, 0.3], py_norm=[0.5, 0.6, 0.8],
    zeta=[0, 0.1, -0.1], delta=[1e-4, 0., -1e-4])

# Compute normalized coordinates
tw = line.twiss()
norm_coord = tw.get_normalized_coordinates(particles, nemitt_x=2.5e-6,
                                           nemitt_y=1e-6)

# Check that the computed normalized coordinates are correct
assert np.allclose(norm_coord['x_norm'], [-1, 0, 0.5], atol=5e-14, rtol=0)
assert np.allclose(norm_coord['y_norm'], [0.3, -0.2, 0.2], atol=5e-14, rtol=0)
assert np.allclose(norm_coord['px_norm'], [0.1, 0.2, 0.3], atol=5e-14, rtol=0)
assert np.allclose(norm_coord['py_norm'], [0.5, 0.6, 0.8], atol=5e-14, rtol=0)
