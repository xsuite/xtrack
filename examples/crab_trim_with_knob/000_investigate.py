# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import time

import json
import numpy as np

import xobjects as xo
import xtrack as xt
import xpart as xp

context = xo.ContextCpu()

###############
# Load a line #
###############

fname_line_particles = '../../test_data/hllhc15_noerrors_nobb/line_w_knobs_and_particle.json'

with open(fname_line_particles, 'r') as fid:
    input_data = json.load(fid)
line = xt.Line.from_dict(input_data['line'])
line.particle_ref = xp.Particles.from_dict(input_data['particle'])

#################
# Build tracker #
#################

line.build_tracker(_context=context)

n_part = 1
particles0 = line.build_particles(x_norm=np.linspace(-1, 1, n_part),
                                nemitt_x=2.5e-6, nemitt_y=2.5e-6)

num_turns = 1000
particles = particles0.copy()
t1 = time.time()
for i_turn in range(num_turns):
    line.track(particles)
t2 = time.time()
print('Time only track: {:.3f} ms/turn'.format((t2 - t1) / num_turns * 1e3))
t_only_track = (t2-t1)

particles = particles0.copy()
t1 = time.time()
for i_turn in range(num_turns):
    line.vars['on_crab1'] = (num_turns - i_turn)/num_turns
    line.track(particles)
t2 = time.time()
print('Overhead track with knob: {:.3f} ms/turn'.format((t2 - t1 - t_only_track) / num_turns * 1e3))

particles = particles0.copy()
t1 = time.time()
for i_turn in range(num_turns):
    line['acfga.4bl1.b1'].knl[0] = 1e-11*i_turn
    line['acfga.4ar1.b1'].ksl[0] = 1e-11*i_turn
    line['acfga.4br1.b1'].ksl[0] = 1e-11*i_turn
    line['acfga.4al1.b1'].ksl[0] = 1e-11*i_turn
    line['acfga.4al1.b1'].knl[0] = 1e-11*i_turn
    line['acfga.4bl1.b1'].ksl[0] = 1e-11*i_turn
    line['acfga.4ar1.b1'].knl[0] = 1e-11*i_turn
    line['acfga.4br1.b1'].knl[0] = 1e-11*i_turn
    line.track(particles)
t2 = time.time()
print('Overhead track with k trim: {:.3f} ms/turn'.format((t2 - t1 - t_only_track) / num_turns * 1e3))


particles = particles0.copy()
t1 = time.time()
for i_turn in range(num_turns):
    line.vars['vrf400'] = (num_turns - i_turn)/num_turns
    line.vars['vrf400'] = (num_turns - i_turn)/num_turns
    line.track(particles)
t2 = time.time()
print('Overhead track with voltage knob: {:.3f} ms/turn'.format((t2 - t1 - t_only_track) / num_turns * 1e3))
