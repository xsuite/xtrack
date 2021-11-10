import pickle
import json
import pathlib
import numpy as np

import xtrack as xt
import xobjects as xo
import xpart as xp

import xslowtrack as xst

from xobjects.context import available

test_data_folder = pathlib.Path(
        __file__).parent.joinpath('../test_data').absolute()



def test_full_rings(element_by_element=False):
     test_fnames =  [
            test_data_folder.joinpath('lhc_no_bb/line_and_particle.json'),
            test_data_folder.joinpath('hllhc_14/line_and_particle.json'),
            test_data_folder.joinpath('sps_w_spacecharge/'
                                      'line_with_spacecharge_and_particle.json'),
            ]

     tolerances_10_turns = [
                    (1e-9, 3e-11),
                    (1e-9, 3e-11),
                    (2e-8, 7e-9)]
     test_backtracker_flags = [True, False, False]

     for icase, fname_line_particles in enumerate(test_fnames):

        rtol_10turns, atol_10turns = tolerances_10_turns[icase]
        test_backtracker= test_backtracker_flags[icase]

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
            # Get a sequence #
            ##################

            line = xt.Line.from_dict(input_data['line'])

            ##################
            # Build TrackJob #
            ##################
            print('Build tracker...')
            tracker = xt.Tracker(_context=context, line=line)

            ######################
            # Get some particles #
            ######################
            particles = xp.Particles(_context=context, **input_data['particle'])

            #########
            # Track #
            #########
            print('Track a few turns...')
            n_turns = 10
            tracker.track(particles, num_turns=n_turns)

            #######################
            # Check against xline #
            #######################

            testline = xst.TestLine.from_dict(input_data['line'])

            print('Check against ...')
            ip_check = 0
            vars_to_check = ['x', 'px', 'y', 'py', 'zeta', 'delta', 's']
            pyst_part = xst.TestParticles.from_dict(input_data['particle'])
            for _ in range(n_turns):
                testline.track(pyst_part)

            for vv in vars_to_check:
                pyst_value = getattr(pyst_part, vv)[0]
                xt_value = context.nparray_from_context_array(
                                                  getattr(particles, vv))[ip_check]
                passed = np.isclose(xt_value, pyst_value,
                                    rtol=rtol_10turns, atol=atol_10turns)
                print(f'Varable {vv}:\n'
                      f'    pyst:   {pyst_value: .7e}\n'
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

                xl_part = xst.TestParticles(**input_data['particle'])

                for vv in vars_to_check:
                    xl_value = getattr(xl_part, vv)[0]
                    xt_value = context.nparray_from_context_array(
                                                getattr(particles, vv))[ip_check]
                    passed = np.isclose(xt_value, xl_value, rtol=rtol_10turns,
                                        atol=atol_10turns)
                    if not passed and vv=='s':
                        passed = np.isclose(xt_value, xl_value,
                                rtol=rtol_10turns, atol=1e-8)

                    if not passed:
                        print(f'Not passend on backtrack for var {vv}!\n'
                              f'    xl:   {xl_value: .7e}\n'
                              f'    xtrack: {xt_value: .7e}\n')
                        #raise ValueError
                        print('Test passed!')


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

        ##################
        # Get a sequence #
        ##################
        line = xt.Line.from_dict(input_data['line'])

        #################
        # Build Tracker #
        #################
        print('Build tracker...')
        freeze_vars = xp.particles.part_energy_varnames() + ['zeta']
        tracker = xt.Tracker(_context=context,
                    line=line,
                    local_particle_src=xp.gen_local_particle_api(
                                                        freeze_vars=freeze_vars),
                    )

        ######################
        # Get some particles #
        ######################
        particle_ref=xp.Particles(**input_data['particle'])
        particles = xp.assemble_particles(_context=context,
                particle_ref=particle_ref,
                x=np.linspace(-1e-4, 1e-4, 10))

        particles_before_tracking = particles.copy()

        #########
        # Track #
        #########
        print('Track a few turns...')
        n_turns = 10
        tracker.track(particles, num_turns=n_turns)

        for vv in ['psigma', 'delta', 'rpp', 'rvv', 'zeta']:
            vv_before = context.nparray_from_context_array(
                                getattr(particles_before_tracking, vv))
            vv_after= context.nparray_from_context_array(
                                getattr(particles, vv))
            assert np.all(vv_before == vv_after)

        print('Check passed')
