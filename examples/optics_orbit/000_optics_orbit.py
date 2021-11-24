import pathlib
import json
import numpy as np

import xobjects as xo
import xtrack as xt
import xpart as xp


fname_line = '../../test_data/hllhc_14/line_and_particle.json'


##############
# Get a line #
##############

with open(fname_line, 'r') as fid:
     input_data = json.load(fid)
line = xt.Line.from_dict(input_data['line'])

tracker = xt.Tracker(line=line)

particle_ref = xp.Particles.from_dict(input_data['particle'])
particle_co = tracker.find_closed_orbit(particle_ref)

R_matrix = tracker.compute_one_turn_matrix_finite_differences(
                   particle_on_co=particle_co)

eta = -R_matrix[4, 5]/line.get_length() # minus sign comes from z = s-ct
alpha_mom_compaction = eta + 1/particle_ref.gamma0[0]**2



