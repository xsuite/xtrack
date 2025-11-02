# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import xtrack as xt
import xpart as xp
import xobjects as xo

line = xt.load('../../test_data/hllhc15_noerrors_nobb/line_and_particle.json')
line.set_particle_ref('proton', p0c=7e12)

num_particles = 50
monitor_ip5 = xt.ParticlesMonitor(start_at_turn=5, stop_at_turn=15,
                                    num_particles=num_particles)
monitor_ip8 = xt.ParticlesMonitor(start_at_turn=5, stop_at_turn=15,
                                    num_particles=num_particles)
line.insert_element(index='ip5', element=monitor_ip5, name='mymon5')
line.insert_element(index='ip8', element=monitor_ip8, name='mymon8')

particles = xp.generate_matched_gaussian_bunch(line=line,
                                               num_particles=num_particles,
                                               nemitt_x=2.5e-6,
                                               nemitt_y=2.5e-6,
                                               sigma_z=9e-2)

num_turns = 30
monitor = xt.ParticlesMonitor(_context=context,
                              start_at_turn=5, stop_at_turn=15,
                              num_particles=num_particles)
line.track(particles, num_turns=num_turns)

# monitor_ip5 contains the data recorded in before the element 'ip5', while
# monitor_ip8 contains the data recorded in before the element 'ip8'
# The element index at which the recording is made can be inspected in
# monitor_ip5.at_element.