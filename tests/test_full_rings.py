# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import pickle
import json
import pathlib
import numpy as np

import xtrack as xt
import xobjects as xo
import xpart as xp

import ducktrack as dtk

test_data_folder = pathlib.Path(
    __file__).parent.joinpath('../test_data').absolute()


def test_full_rings(tmp_path):
    test_fnames = [
        test_data_folder.joinpath('lhc_no_bb/line_and_particle.json'),
        test_data_folder.joinpath('hllhc_14/line_and_particle.json'),
        test_data_folder.joinpath('sps_w_spacecharge/'
                                  'line_no_spacecharge_and_particle.json'),
    ]

    tolerances_10_turns = [
        (1e-9, 3e-11),
        (1e-9, 3e-11),
        (2e-8, 7e-9)]
    test_backtracker_flags = [True, False, False]

    for icase, fname_line_particles in enumerate(test_fnames):

        rtol_10turns, atol_10turns = tolerances_10_turns[icase]
        test_backtracker = test_backtracker_flags[icase]

        print('Case:', fname_line_particles)
        for context in xo.context.get_test_contexts():
            print(f"Test {context.__class__}")

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
            tracker = xt.Tracker(_context=context, line=line,
                                 reset_s_at_end_turn=False)

            ######################
            # Get some particles #
            ######################
            particles = xp.Particles.from_dict(input_data['particle'],
                                               _context=context)
            #########
            # Track #
            #########
            print('Track a few turns...')
            n_turns = 10
            tracker.track(particles, num_turns=n_turns)

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
                xt_value = context.nparray_from_context_array(
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
                backtracker = tracker.get_backtracker(_context=context)
                backtracker.track(particles, num_turns=n_turns)

                dtk_part = dtk.TestParticles(**input_data['particle']).copy()

                for vv in vars_to_check:
                    dtk_value = getattr(dtk_part, vv)[0]
                    xt_value = context.nparray_from_context_array(
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

            part_co = tracker.find_closed_orbit(particle_co_guess=xp.Particles(
                _context=context,
                p0c=input_data['particle']['p0c']))

            parttest = part_co.copy()
            for _ in range(10):
                tracker.track(parttest)
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
            tracker.to_binary_file(tmp_file_path)
            new_tracker = xt.Tracker.from_binary_file(tmp_file_path)

            assert np.all(new_tracker._buffer.buffer == new_tracker._buffer.buffer)
            if tracker.line._var_management:
                assert tracker._tracker_data.line._var_management_to_dict() == \
                    new_tracker._tracker_data.line._var_management_to_dict()
            else:
                assert tracker.line._var_management is \
                    new_tracker.line._var_management is None


def test_freeze_vars():
    for context in xo.context.get_test_contexts():
        print(f"Test {context.__class__}")

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
        tracker = xt.Tracker(_context=context,
                             line=line,
                             reset_s_at_end_turn=False,
                             )
        tracker.freeze_vars(freeze_vars)

        ######################
        # Get some particles #
        ######################
        particle_ref = xp.Particles.from_dict(input_data['particle'])
        particles = xp.build_particles(_context=context,
                                       particle_ref=particle_ref,
                                       x=np.linspace(-1e-4, 1e-4, 10))

        particles_before_tracking = particles.copy()

        #########
        # Track #
        #########
        print('Track a few turns...')
        n_turns = 10
        tracker.track(particles, num_turns=n_turns)

        for vv in ['ptau', 'delta', 'rpp', 'rvv', 'zeta']:
            vv_before = context.nparray_from_context_array(
                getattr(particles_before_tracking, vv))
            vv_after = context.nparray_from_context_array(
                getattr(particles, vv))
            assert np.all(vv_before == vv_after)

        print('Check passed')
