# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import pathlib
import json
import pickle
import numpy as np

import xtrack as xt
import xobjects as xo
import xpart as xp

import ducktrack as dtk

import time


test_data_folder = pathlib.Path(
        __file__).parent.joinpath('../../test_data').absolute()

fname_line_particles = test_data_folder.joinpath('lhc_no_bb/line_and_particle.json')
rtol_100turns = 1e-9; atol_100turns=5e-11

fname_line_particles = test_data_folder.joinpath(
                                  './lhc_with_bb/line_and_particle.json')
rtol_100turns = 1e-9; atol_100turns=9e-10

#fname_line_particles = test_data_folder.joinpath(
#                          './hllhc_14/line_and_particle.json')
#rtol_100turns = 1e-9; atol_100turns=8e-11

# fname_line_particles = test_data_folder.joinpath(
#                  './sps_w_spacecharge/line_no_spacecharge_and_particle.json')
# rtol_100turns = 1e-9; atol_100turns=3e-11

# fname_line_particles = test_data_folder.joinpath(
#                    './sps_w_spacecharge/line_with_spacecharge_and_particle.json')
# rtol_100turns = 2e-8; atol_100turns=7e-9

short_test = False
num_turns = int(100)

####################
# Choose a context #
####################

#n_part = 8000
#context = xo.ContextCpu(omp_num_threads=8)

n_part = 200
context = xo.ContextCpu(omp_num_threads=0)

# n_part = 20000
# context = xo.ContextCupy()

#n_part = 20000
#context = xo.ContextPyopencl('0.0')

#############
# Load file #
#############

with open(fname_line_particles, 'r') as fid:
    input_data = json.load(fid)

##############
# Get a line #
##############

print('Import line')
line= xt.Line.from_dict(input_data['line'])

##################
# Build TrackJob #
##################

print('Build tracker')
line.build_tracker(_context=context, reset_s_at_end_turn=False)

######################
# Get some particles #
######################

print('Import particles')
part_ref = xp.Particles(**input_data['particle'])

# Go from one particle to many particles

particles = xp.build_particles(_context=context,
    particle_ref=part_ref,
    x=np.arange(-1e-4, 1e-4, 2e-4/n_part),
    y=np.arange(-2e-4, 2e-4, 4e-4/n_part))

#########
# Track #
#########
particles_before_tracking = particles.copy(_context=xo.ContextCpu())
print('Track!')
print(f'context: {line._buffer.context}')
line.track(particles, num_turns=num_turns, time=True)
print(f'Time {(line.time_last_track)*1000:.2f} ms')
print(f'Time {(line.time_last_track)*1e6/num_turns/n_part:.2f} us/part/turn')

###########################
# Check against ducktrack #
###########################

testline = dtk.TestLine.from_dict(input_data['line'])

ip_check = n_part//4*3

print(f'\nTest against ducktrack over {num_turns} turns on particle {ip_check}:')
vars_to_check = ['x', 'px', 'y', 'py', 'zeta', 'delta', 's']
part_dict = particles_before_tracking.to_dict()
part_to_check = {}
for kk, vv in part_dict.items():
    if hasattr(vv, '__iter__') and not kk.startswith('_'):
        part_to_check[kk] = part_dict[kk][ip_check]
    else:
        part_to_check[kk] = part_dict[kk]

dtk_part = dtk.TestParticles(**part_to_check)


for iturn in range(num_turns):
    print(f'turn {iturn}/{num_turns}', end='\r', flush=True)
    testline.track(dtk_part)

for vv in vars_to_check:
    dtk_value = getattr(dtk_part, vv)
    xt_value = context.nparray_from_context_array(
                        getattr(particles, vv)[ip_check])
    passed = np.isclose(xt_value, dtk_value,
                        rtol=rtol_100turns, atol=atol_100turns)
    if not passed:
        print(f'Not passed on var {vv}!\n'
              f'    dtk:    {dtk_value: .7e}\n'
              f'    xtrack: {xt_value: .7e}\n')
        raise ValueError
    else:
        print(f'Passed on var {vv}!\n'
              f'    dtk:    {dtk_value: .7e}\n'
              f'    xtrack: {xt_value: .7e}\n')
