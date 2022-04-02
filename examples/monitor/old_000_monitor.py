import json

import numpy as np

import xtrack as xt
import xpart as xp
import xobjects as xo

#context = xo.ContextPyopencl()
context = xo.ContextCpu()

with open('../../test_data/hllhc15_noerrors_nobb/line_and_particle.json') as f:
    dct = json.load(f)
line = xt.Line.from_dict(dct['line'])
line.particle_ref = xp.Particles.from_dict(dct['particle'])

tracker = line.build_tracker(_context=context)

num_particles = 50
particles0 = xp.generate_matched_gaussian_bunch(tracker=tracker,
                                               num_particles=num_particles,
                                               nemitt_x=2.5e-6,
                                               nemitt_y=2.5e-6,
                                               sigma_z=9e-2)
#####################################
# Save all turns starting from zero #
#####################################

# (particles.at_turn must be 0 at the beginning)
particles = particles0.copy()
num_turns = 500
tracker.track(particles, num_turns=num_turns, turn_by_turn_monitor=True)
# ---> tracker.record_last_track.x (contains the z x coordinates)

mon = tracker.record_last_track

######################################
# Save 10 turns starting from turn 5 #
######################################

# Build a monitor
monitor = xt.ParticlesMonitor(_context=context,
                            start_at_turn=5, stop_at_turn=15,
                            num_particles=num_particles)

# (particles.at_turn must be 0 at the beginning)
particles = particles0.copy()
num_turns = 500
tracker.track(particles, num_turns=num_turns, turn_by_turn_monitor=monitor)

#######################
# Multi-frame monitor #
#######################

# Repeated frames
monitor_multiframe = xt.ParticlesMonitor(_context=context,
                                    start_at_turn=5, stop_at_turn=15,
                                    n_repetitions=3,
                                    repetition_period=100,
                                    num_particles=num_particles)


# (particles.at_turn must be 0 at the beginning)
particles = particles0.copy()
num_turns = 500
tracker.track(particles, num_turns=num_turns,
              turn_by_turn_monitor=monitor_multiframe)

assert np.all(mon.x.shape == np.array([50, 500]))
assert np.all(mon.at_turn[3, :] == np.arange(0, 500))
assert np.all(mon.particle_id[:, 3] == np.arange(0, num_particles))

assert np.all(monitor.x.shape == np.array([50, 10]))
assert np.all(monitor.at_turn[3, :] == np.arange(5, 15))
assert np.all(monitor.particle_id[:, 3] == np.arange(0, num_particles))

assert np.all(monitor_multiframe.x.shape == np.array([3, 50, 10]))
assert np.all(monitor_multiframe.at_turn[1, 3, :] == np.arange(105, 115))
assert np.all(monitor_multiframe.particle_id[2, :, 3] == np.arange(0, num_particles))

