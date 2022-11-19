# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import json
import time
import numpy as np

import xtrack as xt
import xpart as xp

###############
# Load a line #
###############

fname_line_particles = '../../test_data/hllhc15_noerrors_nobb/line_and_particle.json'

with open(fname_line_particles, 'r') as fid:
    input_data = json.load(fid)
line = xt.Line.from_dict(input_data['line'])
line.particle_ref = xp.Particles.from_dict(input_data['particle'])

#################
# Build tracker #
#################

tracker = line.build_tracker()


particles = tracker.build_particles(
    x_norm=np.linspace(-2, 2, 1000), y_norm=0.1, delta=3e-4,
    nemitt_x=2.5e-6, nemitt_y=2.5e-6)

p_no_optimized = particles.copy()
p_optimized = particles.copy()

t1 = time.time()
tracker.track(p_no_optimized)
t2 = time.time()
t_no_optimized = t2-t1

tracker.optimize_for_tracking()

t1 = time.time()
tracker.track(p_optimized)
t2 = time.time()
t_optimized = t2-t1

print(f'Time no optimized {t_no_optimized*1e6/len(p_no_optimized.x)} us/particle')
print(f'Time optimized {t_optimized*1e6/len(p_no_optimized.x)} us/particle')
