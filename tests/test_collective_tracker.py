# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import pathlib
import json
import numpy as np

import xobjects as xo
import xtrack as xt
import xfields as xf
import xpart as xp
from xobjects.test_helpers import for_all_test_contexts

import ducktrack as dtk


@for_all_test_contexts
def test_collective_tracker_indices_one_turn(test_context):
    line = xt.Line(elements=[xt.Drift(length=1) for i in range(8)])

    line.elements[2].iscollective = True
    line.elements[2].move(_context=test_context)

    line.elements[5].iscollective = True
    line.elements[5].move(_context=test_context)

    line.build_tracker(_context=test_context)
    line.reset_s_at_end_turn = False

    particles = xp.Particles(x=0, px=0, _context=test_context)

    line.track(particles)

    assert particles.s[0] == len(line.elements)
    assert particles.at_turn[0] == 1


@for_all_test_contexts
def test_collective_tracker(test_context):
    test_data_folder = pathlib.Path(
        __file__).parent.joinpath('../test_data').absolute()
    path_line = test_data_folder.joinpath('sps_w_spacecharge/'
                               'line_with_spacecharge_and_particle.json')

    ##############
    # Get a line #
    ##############

    with open(path_line, 'r') as fid:
         input_data = json.load(fid)
    line = xt.Line.from_dict(input_data['line'])

    # Replace all spacecharge with xobjects
    _buffer = test_context.new_buffer()
    spch_elements = xf.replace_spacecharge_with_quasi_frozen(
                                    line, _buffer=_buffer)

    # For testing I make them frozen but I leave iscollective=True
    for ee in spch_elements:
        ee.update_mean_x_on_track = False
        ee.update_mean_y_on_track = False
        ee.update_sigma_x_on_track = False
        ee.update_sigma_y_on_track = False
        assert ee.iscollective

    #################
    # Build Tracker #
    #################
    print('Build tracker...')
    line.build_tracker(_buffer=_buffer)
    line.reset_s_at_end_turn = False

    assert line.iscollective

    ######################
    # Get some particles #
    ######################
    particles = xp.Particles(_context=test_context, **input_data['particle'])

    #########
    # Track #
    #########

    print('Track a few turns...')
    n_turns = 10
    line.track(particles, num_turns=n_turns,
                  turn_by_turn_monitor=True)

    assert line.record_last_track.x.shape == (1, 10)

    ###########################
    # Check against ducktrack #
    ###########################

    testline = dtk.TestLine.from_dict(input_data['line'])

    print('Check against ducktrack ...')
    ip_check = 0
    vars_to_check = ['x', 'px', 'y', 'py', 'zeta', 'delta', 's']
    dtk_part = dtk.TestParticles.from_dict(input_data['particle'])
    for _ in range(n_turns):
        testline.track(dtk_part)

    for vv in vars_to_check:
        dtk_value = getattr(dtk_part, vv)[0]
        xt_value = test_context.nparray_from_context_array(getattr(particles, vv))[ip_check]
        passed = np.isclose(xt_value, dtk_value, rtol=2e-8, atol=7e-9)
        print(f'Check var {vv}:\n'
              f'    dtk:    {dtk_value: .7e}\n'
              f'    xtrack: {xt_value: .7e}\n')
        if not passed:
            raise ValueError

@for_all_test_contexts
def test_get_non_collective_line(test_context):

    line = xt.Line(
        elements=[xt.Drift(length=1) for i in range(8)],
        element_names=[f'e{i}' for i in range(8)]
    )
    line['e3'].iscollective = True
    e3_buffer = line['e3']._buffer
    e3 = line['e3']

    try:
        line.iscollective
    except RuntimeError:
        pass
    else:
        raise ValueError('This should have failed')

    try:
        line._buffer
    except RuntimeError:
        pass
    else:
        raise ValueError('This should have failed')

    line.build_tracker(_context=test_context)

    assert line.iscollective == True
    assert line['e0']._buffer is line._buffer
    assert line['e7']._buffer is line._buffer
    assert line['e3']._buffer is not line._buffer
    assert line['e3']._buffer is e3_buffer
    assert line['e3'] is e3
    assert line.tracker.line is line

    nc_line = line._get_non_collective_line()

    # Check that the original line is untouched
    assert line.iscollective == True
    assert line['e0']._buffer is line._buffer
    assert line['e7']._buffer is line._buffer
    assert line['e3']._buffer is not line._buffer
    assert line['e3']._buffer is e3_buffer
    assert line['e3'] is e3
    assert line.tracker.line is line

    assert nc_line.iscollective == False
    assert nc_line._buffer is line._buffer
    assert nc_line['e0']._buffer is line._buffer
    assert nc_line['e7']._buffer is line._buffer
    assert nc_line['e3']._buffer is line._buffer
    assert nc_line['e3'] is not e3
    assert nc_line['e0'] is line['e0']
    assert nc_line['e7'] is line['e7']
    assert nc_line.tracker.line is nc_line

    assert np.allclose(nc_line.get_s_elements(), line.get_s_elements(),
                    rtol=0, atol=1e-15)

    assert nc_line.tracker is not line.tracker
    assert nc_line.tracker._tracker_data_cache is line.tracker._tracker_data_cache
    assert line.tracker._track_kernel is nc_line.tracker._track_kernel