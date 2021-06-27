import pickle
import json
import pathlib
import numpy as np

import xobjects as xo
import xtrack as xt
import xline as xl

from make_short_line import make_short_line

short_test = False # Short line (5 elements)

test_data_folder = pathlib.Path(
        __file__).parent.joinpath('../../test_data').absolute()

fname_line_particles = test_data_folder.joinpath('lhc_no_bb/line_and_particle.json')
rtol_10turns = 1e-9; atol_10turns=1e-11

# fname_line_particles = test_data_folder.joinpath(
#                                 './lhc_with_bb/line_and_particle.json')
# rtol_10turns = 1e-9; atol_10turns=1e-11

# fname_line_particles = test_data_folder.joinpath(
#                         './hllhc_14/line_and_particle.json')
# rtol_10turns = 1e-9; atol_10turns=1e-11

# fname_line_particles = test_data_folder.joinpath(
#                  './sps_w_spacecharge/line_without_spacecharge_and_particle.json')
# rtol_10turns = 1e-9; atol_10turns=1e-11

# fname_line_particles = test_data_folder.joinpath(
#                    './sps_w_spacecharge/line_with_spacecharge_and_particle.json')
# rtol_10turns = 2e-8; atol_10turns=7e-9

####################
# Choose a context #
####################

context = xo.ContextCpu()
#context = xo.ContextCupy()
#context = xo.ContextPyopencl('0.0')

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

sequence = xl.Line.from_dict(input_data['line'])
if short_test:
    sequence = make_short_line(sequence)

##################
# Build TrackJob #
##################
print('Build tracker...')
tracker = xt.Tracker(_context=context,
            sequence=sequence,
            particles_class=xt.Particles,
            local_particle_src=None,
            save_source_as='source.c')

######################
# Get some particles #
######################
particles = xt.Particles(_context=context, **input_data['particle'])

#########
# Track #
#########
print('Track a few turns...')
n_turns = 10
tracker.track(particles, num_turns=n_turns)

#######################
# Check against xline #
#######################
print('Check against xline...')
ip_check = 0
vars_to_check = ['x', 'px', 'y', 'py', 'zeta', 'delta', 's']
xl_part = xl.Particles.from_dict(input_data['particle'])
for _ in range(n_turns):
    sequence.track(xl_part)

for vv in vars_to_check:
    xl_value = getattr(xl_part, vv)
    xt_value = context.nparray_from_context_array(getattr(particles, vv))[ip_check]
    passed = np.isclose(xt_value, xl_value, rtol=rtol_10turns, atol=atol_10turns)
    if not passed:
        print(f'Not passend on var {vv}!\n'
              f'    xl:   {xl_value: .7e}\n'
              f'    xtrack: {xt_value: .7e}\n')
        raise ValueError

##############
# Check  ebe #
##############
print('Check element-by-element against xline...')
xl_part = xl.Particles.from_dict(input_data['particle'])
vars_to_check = ['x', 'px', 'y', 'py', 'zeta', 'delta', 's']
problem_found = False
for ii, (eexl, nn) in enumerate(zip(sequence.elements, sequence.element_names)):
    vars_before = {vv :getattr(xl_part, vv) for vv in vars_to_check}
    pp_dict = xt.pyparticles_to_xtrack_dict(xl_part)
    particles.set_particle(ip_check, **pp_dict)

    tracker.track(particles, ele_start=ii, num_elements=1)

    eexl.track(xl_part)
    for vv in vars_to_check:
        xl_change = getattr(xl_part, vv) - vars_before[vv]
        xt_change = context.nparray_from_context_array(
                getattr(particles, vv))[ip_check] -vars_before[vv]
        passed = np.isclose(xt_change, xl_change, rtol=1e-10, atol=5e-14)
        if not passed:
            problem_found = True
            print(f'Not passend on var {vv}!\n'
                  f'    xl:   {xl_change: .7e}\n'
                  f'    xtrack: {xt_change: .7e}\n')
            break

    if not passed:
        print(f'\nelement {nn}')
        break
    else:
        print(f'Check passed for element: {nn}              ', end='\r', flush=True)


if not problem_found:
    print('\nAll passed on context:')
    print(context)

