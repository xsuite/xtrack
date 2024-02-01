# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import json
import time
import numpy as np

import xtrack as xt
import xobjects as xo

num_turns = 100
num_particles = 1000

# Select a running mode
context = xo.ContextCpu()                       # Serial single core
# context = xo.ContextCpu(omp_num_threads=4)      # Serial multi core (4 threads)
# context = xo.ContextCpu(omp_num_threads='auto') # Serial multi core (automatic n. threads)

#################################
# Load a line and build tracker #
#################################

line = xt.Line.from_json(
    '../../test_data/hllhc15_noerrors_nobb/line_w_knobs_and_particle.json')
line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, p0c=7e12)

line.build_tracker(_context=context, compile=False)

###########################
# Generate some particles #
###########################

particles = line.build_particles(
    x_norm=np.linspace(-2, 2, num_particles), y_norm=0.1, delta=3e-4,
    nemitt_x=2.5e-6, nemitt_y=2.5e-6)

####################
# Optimize tracker #
####################

line.optimize_for_tracking()

###########################
# Track with optimization #
###########################

print('Start tracking')
t1 = time.time()
line.track(particles, num_turns=num_turns, time=True)
tracking_time = line.time_last_track
t2=time.time()

particles.move(_context = xo.ContextCpu())

assert len(particles.state) == num_particles
assert np.all(particles.state == 1)

print(f'Tracking time [us/p/turn]: {tracking_time*1e6/num_particles/num_turns:.2f}')
