# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import json
import time
import numpy as np

import xtrack as xt
import xpart as xp

#################################
# Load a line and build tracker #
#################################

fname_line_particles = '../../test_data/hllhc15_noerrors_nobb/line_and_particle.json'

with open(fname_line_particles, 'r') as fid:
    input_data = json.load(fid)
line = xt.Line.from_dict(input_data['line'])
line.particle_ref = xp.Particles.from_dict(input_data['particle'])

line.build_tracker()

###########################
# Generate some particles #
###########################

particles = line.build_particles(
    x_norm=np.linspace(-2, 2, 1000), y_norm=0.1, delta=3e-4,
    nemitt_x=2.5e-6, nemitt_y=2.5e-6)

p_no_optimized = particles.copy()
p_optimized = particles.copy()

##############################
# Track without optimization #
##############################

num_turns = 10
line.track(p_no_optimized, num_turns=num_turns, time=True)
t_not_optimized = line.time_last_track

####################
# Optimize tracker #
####################

line.optimize_for_tracking()

# This performs the following actions (physics model is unchanged):
# - Disables xdeps expressions
# - Removes inactive multipoles
# - Merges consecutive multipoles
# - Removes drifts with zero length
# - Merges consecutive drifts

###########################
# Track with optimization #
###########################

line.track(p_optimized, num_turns=num_turns, time=True)
t_optimized = line.time_last_track

#################
# Compare times #
#################

num_particles = len(p_no_optimized.x)
print(f'Time not optimized {t_not_optimized*1e6/num_particles/num_turns:.1f} us/part/turn')
print(f'Time optimized {t_optimized*1e6/num_particles/num_turns:.1f} us/part/turn')

###################################
# Check that result are identical #
###################################

assert np.all(p_no_optimized.state == 1)
assert np.all(p_optimized.state == 1)

assert np.allclose(p_no_optimized.x, p_optimized.x, rtol=0, atol=1e-14)
assert np.allclose(p_no_optimized.y, p_optimized.y, rtol=0, atol=1e-14)
assert np.allclose(p_no_optimized.px, p_optimized.px, rtol=0, atol=1e-14)
assert np.allclose(p_no_optimized.py, p_optimized.py, rtol=0, atol=1e-14)
assert np.allclose(p_no_optimized.zeta, p_optimized.zeta, rtol=0, atol=1e-14)
assert np.allclose(p_no_optimized.delta, p_optimized.delta, rtol=0, atol=1e-14)