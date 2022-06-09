import json

import xtrack as xt
import xpart as xp
import xobjects as xo

context = xo.ContextCpu()

with open('../../test_data/hllhc15_noerrors_nobb/line_and_particle.json') as f:
    dct = json.load(f)
line = xt.Line.from_dict(dct['line'])
line.particle_ref = xp.Particles.from_dict(dct['particle'])

tracker = line.build_tracker()

num_particles = 50
particles = xp.generate_matched_gaussian_bunch(tracker=tracker,
                                               num_particles=num_particles,
                                               nemitt_x=2.5e-6,
                                               nemitt_y=2.5e-6,
                                               sigma_z=9e-2)

num_turns = 30
tracker.track(particles, num_turns=num_turns,
              turn_by_turn_monitor=True # <--
             )
# tracker.record_last_track contains the measured data. For example,
# tracker.record_last_track.x contains the x coordinate for all particles
# and all turns, e.g. tracker.record_last_track.x[3, 5] for the particle
# having particle_id = 3 and for the turn number 5.
