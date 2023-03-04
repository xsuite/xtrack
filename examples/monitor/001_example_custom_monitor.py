# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import json

import xtrack as xt
import xpart as xp
import xobjects as xo

context = xo.ContextCpu()

with open('../../test_data/hllhc15_noerrors_nobb/line_and_particle.json') as f:
    dct = json.load(f)
line = xt.Line.from_dict(dct['line'])
line.particle_ref = xp.Particles.from_dict(dct['particle'])

line.build_tracker()

num_particles = 50
particles = xp.generate_matched_gaussian_bunch(line=line,
                                               num_particles=num_particles,
                                               nemitt_x=2.5e-6,
                                               nemitt_y=2.5e-6,
                                               sigma_z=9e-2)

num_turns = 30
monitor = xt.ParticlesMonitor(_context=context,
                              start_at_turn=5, stop_at_turn=15,
                              num_particles=num_particles)
line.track(particles, num_turns=num_turns,
              turn_by_turn_monitor=monitor
             )
# line.record_last_track contains the measured data. For example,
# line.record_last_track.x contains the x coordinate for all particles
# and the selected turns, e.g. line.record_last_track.x[3, 5] gives the
# x coordinates for the particle having particle_id = 3 and for the fifth
# recorded turn. The turn indeces that are recorded can be inspected in
# line.record_last_track.at_turn
