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

tracker = line.build_tracker()

particles = tracker.build_particles(
    nemitt_x=2.5e-6, nemitt_y=1e-6,
    x_norm=[-1, 0, 0.5], y_norm=[0.3, -0.2, 0.2],
    px_norm=[0.1, 0.2, 0.3], py_norm=[0.5, 0.6, 0.8],
    zeta=[0, 0.1, -0.1], delta=[1e-4, 0., -1e-4])

tw = tracker.twiss()

norm_coord = tw.get_normalized_coordinates(particles, nemitt_x=2.5e-6,
                                           nemitt_y=1e-6)

assert np.allclose(norm_coord['x_norm'], [-1, 0, 0.5], atol=5e-14, rtol=0)
assert np.allclose(norm_coord['y_norm'], [0.3, -0.2, 0.2], atol=5e-14, rtol=0)
assert np.allclose(norm_coord['px_norm'], [0.1, 0.2, 0.3], atol=5e-14, rtol=0)
assert np.allclose(norm_coord['py_norm'], [0.5, 0.6, 0.8], atol=5e-14, rtol=0)


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

assert np.allclose(norm_coord1['x_norm'], [-1, 0, 0.5], atol=5e-14, rtol=0)
assert np.allclose(norm_coord1['y_norm'], [0.3, -0.2, 0.2], atol=5e-14, rtol=0)
assert np.allclose(norm_coord1['px_norm'], [0.1, 0.2, 0.3], atol=5e-14, rtol=0)
assert np.allclose(norm_coord1['py_norm'], [0.5, 0.6, 0.8], atol=5e-14, rtol=0)

# Check computation at different locations

particles2 = tracker.build_particles(at_element='s.ds.r3.b1',
    _capacity=10,
    nemitt_x=2.5e-6, nemitt_y=1e-6,
    x_norm=[-1, 0, 0.5], y_norm=[0.3, -0.2, 0.2],
    px_norm=[0.1, 0.2, 0.3], py_norm=[0.5, 0.6, 0.8],
    zeta=[0, 0.1, -0.1], delta=[1e-4, 0., -1e-4])

particles3 = tracker.build_particles(at_element='s.ds.r7.b1',
    _capacity=10,
    nemitt_x=2.5e-6, nemitt_y=1e-6,
    x_norm=[-1, 0, 0.5], y_norm=[0.3, -0.2, 0.2],
    px_norm=[0.1, 0.2, 0.3], py_norm=[0.5, 0.6, 0.8],
    zeta=[0, 0.1, -0.1], delta=[1e-4, 0., -1e-4])

particles23 = xp.Particles.merge([particles2, particles3])

norm_coord23 = tw1.get_normalized_coordinates(particles23, nemitt_x=2.5e-6,
                                              nemitt_y=1e-6)

assert particles23._capacity == 20
assert np.allclose(norm_coord23['x_norm'][:3], [-1, 0, 0.5], atol=5e-14, rtol=0)
assert np.allclose(norm_coord23['x_norm'][3:6], [-1, 0, 0.5], atol=5e-14, rtol=0)
assert np.allclose(norm_coord23['x_norm'][6:], xp.particles.LAST_INVALID_STATE)
assert np.allclose(norm_coord23['y_norm'][:3], [0.3, -0.2, 0.2], atol=5e-14, rtol=0)
assert np.allclose(norm_coord23['y_norm'][3:6], [0.3, -0.2, 0.2], atol=5e-14, rtol=0)
assert np.allclose(norm_coord23['y_norm'][6:], xp.particles.LAST_INVALID_STATE)
assert np.allclose(norm_coord23['px_norm'][:3], [0.1, 0.2, 0.3], atol=1e-12, rtol=0)
assert np.allclose(norm_coord23['px_norm'][3:6], [0.1, 0.2, 0.3], atol=1e-12, rtol=0)
assert np.allclose(norm_coord23['px_norm'][6:], xp.particles.LAST_INVALID_STATE)
assert np.allclose(norm_coord23['py_norm'][:3], [0.5, 0.6, 0.8], atol=1e-12, rtol=0)
assert np.allclose(norm_coord23['py_norm'][3:6], [0.5, 0.6, 0.8], atol=1e-12, rtol=0)
assert np.allclose(norm_coord23['py_norm'][6:], xp.particles.LAST_INVALID_STATE)

assert np.all(particles23.at_element[:3] == line.element_names.index('s.ds.r3.b1'))
assert np.all(particles23.at_element[3:6] == line.element_names.index('s.ds.r7.b1'))
assert np.all(particles23.at_element[6:] == xp.particles.LAST_INVALID_STATE)
