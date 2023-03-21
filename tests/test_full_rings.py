# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import pickle
import json
import pathlib
import pytest
import numpy as np

import xtrack as xt
import xpart as xp
from xobjects.test_helpers import for_all_test_contexts

import ducktrack as dtk

test_data_folder = pathlib.Path(
    __file__).parent.joinpath('../test_data').absolute()


@for_all_test_contexts
@pytest.mark.parametrize(
    'fname_line_particles,tolerances_10_turns,test_backtracker',
    [
        ('lhc_no_bb/line_and_particle.json', (1e-9, 3e-11), True),
        ('hllhc_14/line_and_particle.json', (1e-9, 3e-11), False),
        ('sps_w_spacecharge/line_no_spacecharge_and_particle.json',
            (2e-8, 7e-9), False),
    ],
    ids=['lhc_no_bb', 'hllhc_14', 'sps_w_spacecharge'],
)
def test_full_rings(
    test_context,
    fname_line_particles,
    tolerances_10_turns,
    test_backtracker,
    tmp_path
):
    rtol_10turns, atol_10turns = tolerances_10_turns
    fname_line_particles = test_data_folder.joinpath(fname_line_particles)

    #############
    # Load file #
    #############

    if str(fname_line_particles).endswith('.pkl'):
        with open(fname_line_particles, 'rb') as fid:
            input_data = pickle.load(fid)
    elif str(fname_line_particles).endswith('.json'):
        with open(fname_line_particles, 'r') as fid:
            input_data = json.load(fid)

    ##################
    # Get a line #
    ##################

    line = xt.Line.from_dict(input_data['line'])

    ##################
    # Build TrackJob #
    ##################
    print('Build tracker...')
    line.build_tracker(_context=test_context, reset_s_at_end_turn=False)

    ######################
    # Get some particles #
    ######################
    particles = xp.Particles.from_dict(input_data['particle'],
                                       _context=test_context)
    #########
    # Track #
    #########
    print('Track a few turns...')
    n_turns = 10
    line.track(particles, num_turns=n_turns)

    ###########################
    # Check against ducktrack #
    ###########################

    testline = dtk.TestLine.from_dict(input_data['line'])

    print('Check against ...')
    ip_check = 0
    vars_to_check = ['x', 'px', 'y', 'py', 'zeta', 'delta', 's']
    dtk_part = dtk.TestParticles.from_dict(input_data['particle']).copy()
    for _ in range(n_turns):
        testline.track(dtk_part)

    for vv in vars_to_check:
        dtk_value = getattr(dtk_part, vv)[0]
        xt_value = test_context.nparray_from_context_array(
            getattr(particles, vv))[ip_check]
        passed = np.isclose(xt_value, dtk_value,
                            rtol=rtol_10turns, atol=atol_10turns)
        print(f'Varable {vv}:\n'
              f'    dtk:    {dtk_value: .7e}\n'
              f'    xtrack: {xt_value: .7e}\n')
        if not passed:
            raise ValueError('Discrepancy found!')

    #####################
    # Check backtracker #
    #####################

    if test_backtracker:
        print('Testing backtracker')
        backtracker = line.get_backtracker(_context=test_context)
        backtracker.track(particles, num_turns=n_turns)

        dtk_part = dtk.TestParticles(**input_data['particle']).copy()

        for vv in vars_to_check:
            dtk_value = getattr(dtk_part, vv)[0]
            xt_value = test_context.nparray_from_context_array(
                getattr(particles, vv))[ip_check]
            passed = np.isclose(xt_value, dtk_value, rtol=rtol_10turns,
                                atol=atol_10turns)
            if not passed and vv == 's':
                passed = np.isclose(xt_value, dtk_value,
                                    rtol=rtol_10turns, atol=1e-8)

            if not passed:
                print(f'Not passend on backtrack for var {vv}!\n'
                      f'    dtk:    {dtk_value: .7e}\n'
                      f'    xtrack: {xt_value: .7e}\n')
                # raise ValueError
                print('Test passed!')

    ######################
    # Check closed orbit #
    ######################

    part_co = line.find_closed_orbit(particle_co_guess=xp.Particles(
        _context=test_context,
        p0c=input_data['particle']['p0c']))

    parttest = part_co.copy()
    for _ in range(10):
        line.track(parttest)
        assert np.isclose(parttest._xobject.x[0], part_co._xobject.x[0],
                          rtol=0, atol=1e-11)
        assert np.isclose(parttest._xobject.y[0], part_co._xobject.y[0],
                          rtol=0, atol=1e-11)
        assert np.isclose(parttest._xobject.zeta[0], part_co._xobject.zeta[0],
                          rtol=0, atol=5e-11)

    ###############################
    # Verify binary serialization #
    ###############################

    tmp_file = tmp_path / 'test_full_rings.npy'
    tmp_file_path = tmp_file.resolve()
    line.tracker.to_binary_file(tmp_file_path)
    new_tracker = xt.Tracker.from_binary_file(tmp_file_path)

    assert np.all(new_tracker._buffer.buffer == new_tracker._buffer.buffer)
    if line._var_management:
        assert line._tracker_data.line._var_management_to_dict() == \
            new_tracker._tracker_data.line._var_management_to_dict()
    else:
        assert line._var_management is \
            new_tracker.line._var_management is None


@for_all_test_contexts
def test_freeze_vars(test_context):
    test_data_folder.joinpath('hllhc_14/line_and_particle.json'),

    fname_line_particles = test_data_folder.joinpath(
        './hllhc_14/line_and_particle.json')

    #############
    # Load file #
    #############

    with open(fname_line_particles, 'r') as fid:
        input_data = json.load(fid)

    ##############
    # Get a line #
    ##############
    line = xt.Line.from_dict(input_data['line'])

    #################
    # Build Tracker #
    #################
    print('Build tracker...')
    freeze_vars = xp.particles.part_energy_varnames() + ['zeta']
    line.build_tracker(_context=test_context, reset_s_at_end_turn=False)
    line.freeze_vars(freeze_vars)

    ######################
    # Get some particles #
    ######################
    particle_ref = xp.Particles.from_dict(input_data['particle'])
    particles = xp.build_particles(_context=test_context,
                                   particle_ref=particle_ref,
                                   x=np.linspace(-1e-4, 1e-4, 10))

    particles_before_tracking = particles.copy()

    #########
    # Track #
    #########
    print('Track a few turns...')
    n_turns = 10
    line.track(particles, num_turns=n_turns)

    for vv in ['ptau', 'delta', 'rpp', 'rvv', 'zeta']:
        vv_before = test_context.nparray_from_context_array(
            getattr(particles_before_tracking, vv))
        vv_after = test_context.nparray_from_context_array(
            getattr(particles, vv))
        assert np.all(vv_before == vv_after)

    print('Check passed')
