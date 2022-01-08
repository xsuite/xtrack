import json
import numpy as np

import xobjects as xo
import xtrack as xt
import xpart as xp

###############
# Load a line #
###############

fname_line_particles = '../../test_data/hllhc15_noerrors_nobb/line_and_particle.json'
#fname_line_particles = '../../test_data/sps_w_spacecharge/line_no_spacecharge_and_particle.json' #!skip-doc

with open(fname_line_particles, 'r') as fid:
    input_data = json.load(fid)
line = xt.Line.from_dict(input_data['line'])
particle_ref = xp.Particles.from_dict(input_data['particle'])

#################
# Build tracker #
#################

tracker = xt.Tracker(line=line)

particles = xp.build_particles(particle_ref=particle_ref, x=np.zeros(1000))
tracker.track(particles, turn_by_turn_monitor='ONE_TURN_EBE')

#newelements = []
#newenames = []
#mm_src = xt.ParticlesMonitor(
#        _buffer=tracker._buffer,
#        start_at_turn=0,
#        stop_at_turn=1,
#        num_particles=1000
#    )
#for ii, (ee, nn) in enumerate(
#                        zip(tracker.line.elements, tracker.line.element_names)):
#    mm = mm_src.copy(_buffer=tracker._buffer)
#    newelements.append(mm)
#    newenames.append(f'ebemon_{ii}')
#    newelements.append(ee)
#    newenames.append(nn)
