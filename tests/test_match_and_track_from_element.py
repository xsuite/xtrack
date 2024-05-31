# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import json
import pathlib

import numpy as np

import xobjects as xo
import xpart as xp
import xtrack as xt
from xobjects.test_helpers import for_all_test_contexts

test_data_folder = pathlib.Path(
        __file__).parent.joinpath('../test_data').absolute()


@for_all_test_contexts
def test_match_and_track_from_element(test_context):
    # Load machine model (from pymask)
    filename = test_data_folder.joinpath('lhc_no_bb/line_and_particle.json')
    with open(filename, 'r') as fid:
        input_data = json.load(fid)
    line = xt.Line.from_dict(input_data['line'])
    line.build_tracker(_context=test_context)
    line.reset_s_at_end_turn = False

    assert not line.iscollective
    line.particle_ref = xp.Particles.from_dict(input_data['particle'])

    # Check matching of a one-sigma circle in ip2
    r_sigma = 1
    theta = np.linspace(0, 2*np.pi, 1000)

    at_element = 'ip2'
    particles = line.build_particles(
                    x_norm=r_sigma*np.cos(theta), px_norm=r_sigma*np.sin(theta),
                    nemitt_x=2.5e-6, nemitt_y=2.5e-6,
                    at_element=at_element)

    tw = line.twiss(at_elements=[at_element])

    particles.move(_context=xo.context_default) # To easily do the checks with numpy
    xo.assert_allclose(
        np.sqrt(tw['betx'][0]*2.5e-6/particles.beta0[0]/particles.gamma0[0]),
        np.max(np.abs(particles.x - np.mean(particles.x))), rtol=1e-3, atol=0)
    particles.move(_context=test_context)

    # Check that tracking starts from the right place
    line.track(particles, turn_by_turn_monitor='ONE_TURN_EBE')
    mon = line.record_last_track
    i_ele_start = line.element_names.index(at_element)
    assert np.all(mon.at_element[:, :i_ele_start] == 0)
    assert np.all(mon.at_element[:, i_ele_start] == i_ele_start)
    assert np.all(mon.at_element[:, -1] == len(line.element_names))

    # Check that distribution is matched at the end of the turn
    tw0 = line.twiss(at_elements=[0])
    particles.move(_context=xo.context_default)
    xo.assert_allclose(
        np.sqrt(tw0['betx'][0]*2.5e-6/particles.beta0[0]/particles.gamma0[0]),
        np.max(np.abs(particles.x - np.mean(particles.x))), rtol=2e-3, atol=0)

    # Check multiple turns
    at_element = 'ip2'
    particles = line.build_particles(
                    x_norm=r_sigma*np.cos(theta), px_norm=r_sigma*np.sin(theta),
                    nemitt_x=2.5e-6, nemitt_y=2.5e-6,
                    at_element=at_element)

    tw = line.twiss(at_elements=[at_element])

    particles.move(_context=xo.context_default)
    xo.assert_allclose(
        np.sqrt(tw['betx'][0]*2.5e-6/particles.beta0[0]/particles.gamma0[0]),
        np.max(np.abs(particles.x - np.mean(particles.x))), rtol=1e-3, atol=0)
    particles.move(_context=test_context)

    line.track(particles, num_turns=3)

    tw0 = line.twiss(at_elements=[0])
    particles.move(_context=xo.context_default)
    xo.assert_allclose(
        np.sqrt(tw0['betx'][0]*2.5e-6/particles.beta0[0]/particles.gamma0[0]),
        np.max(np.abs(particles.x - np.mean(particles.x))), rtol=2e-3, atol=0)
    assert np.all(particles.at_turn==3)
    xo.assert_allclose(particles.s, 3*line.get_length(), rtol=0, atol=1e-7)

    # Check collective case
    line_w_collective = xt.Line.from_dict(input_data['line'])
    for ip in range(8):
        line_w_collective.element_dict[f'ip{ip+1}'].iscollective = True
        line_w_collective.element_dict[f'ip{ip+1}'].move(_context=test_context)
    line_w_collective.build_tracker(_context=test_context)
    line_w_collective.reset_s_at_end_turn = False

    assert line_w_collective.iscollective
    line_w_collective.particle_ref = xp.Particles.from_dict(input_data['particle'])
    assert len(line_w_collective.tracker._parts) == 16

    at_element = 'ip2'
    particles = line_w_collective.build_particles(
                    x_norm=r_sigma*np.cos(theta), px_norm=r_sigma*np.sin(theta),
                    nemitt_x=2.5e-6, nemitt_y=2.5e-6,
                    at_element=at_element)

    tw = line_w_collective.twiss(at_elements=[at_element])

    particles.move(_context=xo.context_default)
    xo.assert_allclose(
        np.sqrt(tw['betx'][0]*2.5e-6/particles.beta0[0]/particles.gamma0[0]),
        np.max(np.abs(particles.x - np.mean(particles.x))), rtol=1e-3, atol=0)
    particles.move(_context=test_context)

    line_w_collective.track(particles, num_turns=3)

    tw0 = line_w_collective.twiss(at_elements=[0])
    particles.move(_context=xo.context_default)
    xo.assert_allclose(
        np.sqrt(tw0['betx'][0]*2.5e-6/particles.beta0[0]/particles.gamma0[0]),
        np.max(np.abs(particles.x - np.mean(particles.x))), rtol=3e-3, atol=0)
    assert np.all(particles.at_turn==3)
    xo.assert_allclose(particles.s, 3*line_w_collective.get_length(), rtol=0, atol=1e-7)

    # Check match_at_s
    at_element = 'ip6'
    particles = line_w_collective.build_particles(
                    x_norm=r_sigma*np.cos(theta), px_norm=r_sigma*np.sin(theta),
                    nemitt_x=2.5e-6, nemitt_y=2.5e-6,
                    at_element=at_element,
                    match_at_s=line_w_collective.get_s_position('ip6') + 100
                    )

    tw = line_w_collective.twiss(at_elements=[at_element])

    particles.move(_context=xo.context_default)
    xo.assert_allclose(
        np.sqrt(tw['betx'][0]*2.5e-6/particles.beta0[0]/particles.gamma0[0]),
        np.max(np.abs(particles.x - np.mean(particles.x))), rtol=1e-3, atol=0)
    xo.assert_allclose(particles.s, tw['s'][0], atol=1e-8, rtol=0)

    phasex_first_part = np.angle(particles.x[0] / np.sqrt(tw['betx'][0]) -
                1j*(particles.x[0]  * tw['alfx'][0] / np.sqrt(tw['betx'][0]) +
                        particles.px[0] * np.sqrt(tw['betx'][0])))

    mu_at_s = line_w_collective.twiss(
        at_s=line_w_collective.get_s_position('ip6') + 100)['mux'][0]
    mu_at_element = line_w_collective.twiss(at_elements=[at_element])['mux'][0]

    xo.assert_allclose(phasex_first_part, (mu_at_element - mu_at_s)*2*np.pi,
                    atol=0, rtol=0.02)
    particles.move(_context=test_context)

    line_w_collective.track(particles, num_turns=3)

    tw0 = line_w_collective.twiss(at_elements=[0])
    particles.move(_context=xo.context_default)
    xo.assert_allclose(
        np.sqrt(tw0['betx'][0]*2.5e-6/particles.beta0[0]/particles.gamma0[0]),
        np.max(np.abs(particles.x - np.mean(particles.x))), rtol=2e-3, atol=0)
    assert np.all(particles.at_turn==3)
    xo.assert_allclose(particles.s, 3*line_w_collective.get_length(), rtol=0, atol=1e-7)
