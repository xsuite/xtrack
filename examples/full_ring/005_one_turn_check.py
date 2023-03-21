# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import json
import pathlib
import numpy as np

import xobjects as xo
import xtrack as xt
import xpart as xp

import ducktrack as dtk

test_data_folder = pathlib.Path(
        __file__).parent.joinpath('../../test_data').absolute()

fname_line_particles = test_data_folder.joinpath('lhc_no_bb/line_and_particle.json')
rtol_10turns = 1e-9; atol_10turns=4e-11
test_backtracker=True

#fname_line_particles = test_data_folder.joinpath(
#                                './lhc_with_bb/line_and_particle.json')
#rtol_10turns = 1e-9; atol_10turns=2e-11
#test_backtracker = False

fname_line_particles = test_data_folder.joinpath(
                         './hllhc_14/line_and_particle.json')
rtol_10turns=1e-9; atol_10turns=1e-11 # 2e-10 needed for delta = 1e-3
test_backtracker = False

#fname_line_particles = test_data_folder.joinpath(
#                    './sps_w_spacecharge/line_with_spacecharge_and_particle.json')
#rtol_10turns = 2e-8; atol_10turns=7e-9
#test_backtracker = False

####################
# Choose a context #
####################

context = xo.ContextCpu()
#context = xo.ContextCupy()
#context = xo.ContextPyopencl('0.0')

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
line.build_tracker(_context=context, reset_s_at_end_turn=False)

if test_backtracker:
    backtracker = line.get_backtracker(_context=context)

######################
# Get some particles #
######################
particles = xp.Particles(_context=context, **input_data['particle'])

# To test off momentum one can do the following:
# #particles.delta = 1e-3
# input_data['particle'] = particles.to_dict()

#########
# Track #
#########
print('Track a few turns...')
n_turns = 10
line.track(particles, num_turns=n_turns)

###########################
# Check against ducktrack #
###########################
print('Check against ducktrack...')

testline = dtk.TestLine.from_dict(input_data['line'])

ip_check = 0
vars_to_check = ['x', 'px', 'y', 'py', 'zeta', 'delta', 's']
dtk_part = dtk.TestParticles.from_dict(input_data['particle']).copy()
for _ in range(n_turns):
   testline.track(dtk_part)

for vv in vars_to_check:
    dtk_value = getattr(dtk_part, vv)[0]
    xt_value = context.nparray_from_context_array(getattr(particles, vv))[ip_check]
    passed = np.isclose(xt_value, dtk_value, rtol=rtol_10turns, atol=atol_10turns)

    if not passed:
        print(f'Not passed on var {vv}!\n'
              f'    dtk:   {dtk_value: .7e}\n'
              f'    xtrack: {xt_value: .7e}\n')
        raise ValueError

#####################
# Check backtracker #
#####################

if test_backtracker:
    backtracker.track(particles, num_turns=n_turns)

    dtk_part = dtk.TestParticles.from_dict(input_data['particle']).copy()

    for vv in vars_to_check:
        dtk_value = getattr(dtk_part, vv)[0]
        xt_value = context.nparray_from_context_array(getattr(particles, vv))[ip_check]
        passed = np.isclose(xt_value, dtk_value, rtol=rtol_10turns,
                            atol=atol_10turns)
        if not passed and vv=='s':
            passed = np.isclose(xt_value, dtk_value,
                    rtol=rtol_10turns, atol=1e-8)

        if not passed:
            print(f'Not passend on backtrack for var {vv}!\n'
                  f'    dtk:   {dtk_value: .7e}\n'
                  f'    xtrack: {xt_value: .7e}\n')
            raise ValueError

##############
# Check  ebe #
##############
print('Check element-by-element against ducktrack...')
dtk_part = dtk.TestParticles.from_dict(input_data['particle']).copy()
vars_to_check = ['x', 'px', 'y', 'py', 'zeta', 'delta', 's']
problem_found = False
diffs = []
s_coord = []
for ii, (eedtk, nn) in enumerate(zip(testline.elements, testline.element_names)):
    vars_before = {vv: getattr(dtk_part, vv)[0] for vv in vars_to_check}
    particles = xp.Particles.from_dict(dtk_part.to_dict(), _context=context)

    line.track(particles, ele_start=ii, num_elements=1)

    eedtk.track(dtk_part)
    s_coord.append(dtk_part.s[0])
    diffs.append([])
    for vv in vars_to_check:
        dtk_change = getattr(dtk_part, vv)[0] - vars_before[vv]
        xt_change = (context.nparray_from_context_array(
                getattr(particles, vv))[ip_check]- vars_before[vv])
        passed = np.isclose(xt_change, dtk_change, rtol=1e-10, atol=5e-14)
        if not passed:
            problem_found = True
            print(f'Not passend on var {vv}!\n'
                  f'    dtk:   {dtk_change: .7e}\n'
                  f'    xtrack: {xt_change: .7e}\n')
            break
        diffs[-1].append(np.abs(
            context.nparray_from_context_array(
                getattr(particles, vv))[ip_check] - getattr(dtk_part, vv)[0]))

    if not passed:
        print(f'\nelement {nn}')
        break

    if test_backtracker:
        backtracker.track(particles,
                ele_start=len(line.elements) - ii - 1,
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
for ii_plt, (vv, uu) in enumerate(
        zip(['x', 'px', 'y', 'py', r'$\zeta$', r'$\delta$'],
            ['[m]', '[-]', '[m]', '[-]', '[m]', '[-]'])):
    ax = fig.add_subplot(3, 2, ii_plt+1)
    ax.plot(s_coord, diffs[:, ii_plt])
    ax.set_ylabel('Difference on '+ vv + ' ' + uu)
    ax.set_xlabel('s [m]')
fig.subplots_adjust(hspace=.48)
plt.show()


