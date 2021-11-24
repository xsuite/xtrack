import pathlib
import json
import numpy as np

import xobjects as xo
import xtrack as xt
import xpart as xp
from scipy.constants import c as clight


fname_line = '../../test_data/hllhc_14/line_and_particle.json'


##############
# Get a line #
##############

with open(fname_line, 'r') as fid:
     input_data = json.load(fid)
line = xt.Line.from_dict(input_data['line'])

tracker = xt.Tracker(line=line)

particle_ref = xp.Particles.from_dict(input_data['particle'])

zeta, delta = xp.longitudinal.generate_longitudinal_coordinates(
        num_particles=100000, distribution='gaussian',
        sigma_z=10e-2, particle_ref=particle_ref, tracker=tracker)

assert np.isclose(np.std(zeta), 10e-2, rtol=1e-2, atol=0)
assert np.isclose(np.std(delta), 1.21e-4, rtol=1e-2, atol=0)
