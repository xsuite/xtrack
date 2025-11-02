# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import xtrack as xt
import xpart as xp
import xobjects as xo

context = xo.ContextCpu()

line = xt.load('../../test_data/hllhc15_noerrors_nobb/line_and_particle.json')
line.set_particle_ref('proton', p0c=7e12)

num_particles = 50
particles = xp.generate_matched_gaussian_bunch(line=line,
                                               num_particles=num_particles,
                                               nemitt_x=2.5e-6,
                                               nemitt_y=2.5e-6,
                                               sigma_z=9e-2)

num_turns = 30
line.track(particles, num_turns=num_turns,
              turn_by_turn_monitor=True # <--
             )
# line.record_last_track contains the measured data. For example,
# line.record_last_track.x contains the x coordinate for all particles
# and all turns, e.g. line.record_last_track.x[3, 5] for the particle
# having particle_id = 3 and for the turn number 5.

# Monitor objects can be dumped to a dictionary and loaded back
mon = line.record_last_track
dct = mon.to_dict()
mon2 = xt.ParticlesMonitor.from_dict(dct)
