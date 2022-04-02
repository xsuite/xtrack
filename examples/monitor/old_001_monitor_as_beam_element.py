import json

import xtrack as xt
import xpart as xp

from cpymad.madx import Madx


with open('../../test_data/hllhc15_noerrors_nobb/line_and_particle.json') as f:
    dct = json.load(f)
line = xt.Line.from_dict(dct['line'])
line.particle_ref = xp.Particles.from_dict(dct['particle'])

tracker = line.build_tracker()

num_particles = 50
particles0 = xp.generate_matched_gaussian_bunch(tracker=tracker,
                                               num_particles=num_particles,
                                               nemitt_x=2.5e-6,
                                               nemitt_y=2.5e-6,
                                               sigma_z=9e-2)
#     #####################################
#     # Save all turns starting from zero #
#     #####################################
#     
#     # (particles.at_turn must be 0 at the beginning)
#     particles = particles0.copy()
#     num_turns = 500
#     tracker.track(particles, num_turns=num_turns, turn_by_turn_monitor=True)
#     # ---> tracker.record_last_track.x (contains the z x coordinates)

######################################
# Save 10 turns starting from turn 5 #
######################################

# Build a monitor
monitor = xt.ParticlesMonitor(start_at_turn=5, stop_at_turn=15,
                              num_particles=num_particles)

# (particles.at_turn must be 0 at the beginning)
particles = particles0.copy()
num_turns = 500
tracker.track(particles, num_turns=num_turns, turn_by_turn_monitor=monitor)

#####################################################
# Save 10 turns starting from turn 5 in ip5 and ip8 #
#####################################################

#     # Build a monitor
#     monitor_ip5 = xt.ParticlesMonitor(start_at_turn=5, stop_at_turn=15,
#                                       num_particles=num_particles)
#     monitor_ip8 = xt.ParticlesMonitor(start_at_turn=5, stop_at_turn=15,
#                                       num_particles=num_particles)
#     
#     line_with_monitor = line.copy()
#     line_with_monitor.insert_element(index='ip5', element=monitor_ip5, name='mymon5')
#     line_with_monitor.insert_element(index='ip8', element=monitor_ip8, name='mymon8')
#     
#     tracker_w_monitor = line_with_monitor.build_tracker()
#     
#     particles = particles0.copy()
#     num_turns = 500
#     tracker_w_monitor.track(particles, num_turns=num_turns)
#     
#     #######################
#     # Multi-frame monitor #
#     #######################

# Repeated frames
monitor_multiframe = xt.ParticlesMonitor(start_at_turn=5, stop_at_turn=15,
                                  n_repetitions=2,
                                  repetition_period=100,
                                  num_particles=num_particles)


# (particles.at_turn must be 0 at the beginning)
particles = particles0.copy()
num_turns = 500
tracker.track(particles, num_turns=num_turns,
              turn_by_turn_monitor=monitor_multiframe)