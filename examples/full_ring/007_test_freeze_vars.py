# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import pickle
import json
import pathlib
import numpy as np

import xobjects as xo
import xtrack as xt
import xpart as xp


short_test = False # Short line (5 elements)

test_data_folder = pathlib.Path(
        __file__).parent.joinpath('../../test_data').absolute()

fname_line_particles = test_data_folder.joinpath(
                        './hllhc_14/line_and_particle.json')

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
line.particle_ref = xp.Particles(**input_data['particle'])

#################
# Build Tracker #
#################
print('Build tracker...')
freeze_vars = xp.particles.part_energy_varnames() + ['zeta']
line.build_tracker(_context=context)

line.freeze_longitudinal()

#########
# Twiss #
#########

tw = line.twiss(method='4d')  # <-- Need to choose 4d mode when longitudinal
                                 #     variables are frozen

##################################
# Match a particles distribution #
##################################

particles = xp.build_particles(_context=context, line=line,
                               mode = '4d',  # <--- 4d
                               x_norm=np.linspace(0, 10, 11),
                               nemitt_x=3e-6, nemitt_y=3e-6)

particles_before_tracking = particles.copy()

#########
# Track #
#########
print('Track a few turns...')
n_turns = 10
line.track(particles, num_turns=n_turns)

print('Track again (no compile)')
line.track(particles, num_turns=n_turns)

for vv in ['ptau', 'delta', 'rpp', 'rvv', 'zeta']:
    vv_before = context.nparray_from_context_array(
                        getattr(particles_before_tracking, vv))
    vv_after= context.nparray_from_context_array(
                        getattr(particles, vv))
    assert np.all(vv_before == vv_after)

print('Check passed')
