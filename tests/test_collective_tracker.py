import pathlib
import json
import numpy as np

import xobjects as xo
import xtrack as xt
import xfields as xf
import xpart as xp

import xslowtrack as xst

def test_collective_tracker():

    for context in xo.context.get_test_contexts():
        print(f"Test {context.__class__}")

        test_data_folder = pathlib.Path(
            __file__).parent.joinpath('../test_data').absolute()
        path_sequence = test_data_folder.joinpath('sps_w_spacecharge/'
                                   'line_with_spacecharge_and_particle.json')
        turn_by_turn_monitor = True

        ##################
        # Get a sequence #
        ##################

        with open(path_sequence, 'r') as fid:
             input_data = json.load(fid)
        line = xt.Line.from_dict(input_data['line'])

        # Replace all spacecharge with xobjects
        _buffer = context.new_buffer()
        spch_elements = xf.replace_spaceharge_with_quasi_frozen(
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
        tracker= xt.Tracker(_buffer=_buffer,
                     line=line)

        assert tracker.iscollective
        assert tracker.track == tracker._track_with_collective

        ######################
        # Get some particles #
        ######################
        particles = xp.Particles(_context=context, **input_data['particle'])

        #########
        # Track #
        #########

        print('Track a few turns...')
        n_turns = 10
        tracker.track(particles, num_turns=n_turns,
                      turn_by_turn_monitor=True)

        assert tracker.record_last_track.x.shape == (1, 10)

        #######################
        # Check against xline #
        #######################

        testline = xst.TestLine.from_dict(input_data['line'])

        print('Check against xline ...')
        ip_check = 0
        vars_to_check = ['x', 'px', 'y', 'py', 'zeta', 'delta', 's']
        pyst_part = xst.TestParticles.from_dict(input_data['particle'])
        for _ in range(n_turns):
            testline.track(pyst_part)

        for vv in vars_to_check:
            pyst_value = getattr(pyst_part, vv)[0]
            xt_value = context.nparray_from_context_array(getattr(particles, vv))[ip_check]
            passed = np.isclose(xt_value, pyst_value, rtol=2e-8, atol=7e-9)
            print(f'Check var {vv}:\n'
                  f'    pyst:   {pyst_value: .7e}\n'
                  f'    xtrack: {xt_value: .7e}\n')
            if not passed:
                raise ValueError

