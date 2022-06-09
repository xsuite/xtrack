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
monitor = xt.ParticlesMonitor(_context=context,
                              start_at_turn=5, stop_at_turn=15,
                              num_particles=num_particles)
tracker.track(particles, num_turns=num_turns,
              turn_by_turn_monitor=monitor
             )
# tracker.record_last_track contains the measured data. For example,
# tracker.record_last_track.x contains the x coordinate for all particles
# and the selected turns, e.g. tracker.record_last_track.x[3, 5] gives the
# x coordinates for the particle having particle_id = 3 and for the fifth
# recorded turn. The turn indeces that are recorded can be inspected in
# tracker.record_last_track.at_turn
