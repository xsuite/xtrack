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
rtol_10turns = 1e-9; atol_10turns=4e-11
test_backtracker=True

# fname_line_particles = test_data_folder.joinpath(
#                                 './lhc_with_bb/line_and_particle.json')
# rtol_10turns = 1e-9; atol_10turns=1e-11
# test_backtracker = False

# fname_line_particles = test_data_folder.joinpath(
#                         './hllhc_14/line_and_particle.json')
# rtol_10turns = 1e-9; atol_10turns=1e-11
# test_backtracker = False

# fname_line_particles = test_data_folder.joinpath(
#                     './sps_w_spacecharge/line_with_spacecharge_and_particle.json')
# rtol_10turns = 2e-8; atol_10turns=7e-9
# test_backtracker = False

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

#################
# Build Tracker #
#################
print('Build tracker...')
tracker = xt.Tracker(_context=context,
            sequence=sequence,
            particles_class=xt.Particles,
            save_source_as='source.c',
            )

if test_backtracker:
    backtracker = tracker.get_backtracker(_context=context)

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

#####################
# Check backtracker #
#####################

if test_backtracker:
    backtracker.track(particles, num_turns=n_turns)

    xl_part = xl.Particles.from_dict(input_data['particle'])

    for vv in vars_to_check:
        xl_value = getattr(xl_part, vv)
        xt_value = context.nparray_from_context_array(getattr(particles, vv))[ip_check]
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

##############
# Check  ebe #
##############
print('Check element-by-element against xline...')
xl_part = xl.Particles.from_dict(input_data['particle'])
vars_to_check = ['x', 'px', 'y', 'py', 'zeta', 'delta', 's']
problem_found = False
diffs = []
s_coord = []
for ii, (eexl, nn) in enumerate(zip(sequence.elements, sequence.element_names)):
    vars_before = {vv :getattr(xl_part, vv) for vv in vars_to_check}
    particles.set_particle(ip_check, **xl_part.to_dict())

    tracker.track(particles, ele_start=ii, num_elements=1)

    eexl.track(xl_part)
    s_coord.append(xl_part.s)
    diffs.append([])
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
        diffs[-1].append(np.abs(
            getattr(particles, vv)[ip_check] - getattr(xl_part, vv)))

    if not passed:
        print(f'\nelement {nn}')
        break

    if test_backtracker:
        backtracker.track(particles,
                ele_start=len(tracker.line.elements) - ii - 1,
                num_elements=1)
        for vv in vars_to_check:
            xt_value = context.nparray_from_context_array(
                                        getattr(particles, vv))[ip_check]
            passed = np.isclose(xt_value, vars_before[vv],
                                rtol=1e-10, atol=1e-13)
            if not passed:
                problem_found = True
                print(f'\nNot passend on var {vv}!\n'
                      f'    before: {vars_before[vv]: .7e}\n'
                      f'    xtrack: {xt_value: .7e}\n')
                break
        if not passed:
            print(f'\nelement {nn}')
            break

    print(f'Check passed for element: {nn}              ', end='\r', flush=True)


diffs = np.array(diffs)

if not problem_found:
    print('\nAll passed on context:')
    print(context)

import matplotlib.pyplot as plt
plt.close('all')
fig = plt.figure(1, figsize=(6.4*1.5, 4.8*1.3))
for ii, (vv, uu) in enumerate(
        zip(['x', 'px', 'y', 'py', r'$\zeta$', r'$\delta$'],
            ['[m]', '[-]', '[m]', '[-]', '[m]', '[-]'])):
    ax = fig.add_subplot(3, 2, ii+1)
    ax.plot(s_coord, diffs[:, ii])
    ax.set_ylabel('Difference on '+ vv + ' ' + uu)
    ax.set_xlabel('s [m]')
fig.subplots_adjust(hspace=.48)
plt.show()


