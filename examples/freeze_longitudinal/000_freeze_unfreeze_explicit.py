import json

import xtrack as xt
import xpart as xp

fname_line = '../../test_data/lhc_no_bb/line_and_particle.json'

# import a line
with open(fname_line) as fid:
    line_dict = json.load(fid)

line = xt.Line.from_dict(line_dict)
line.particle_ref = xp.Particles.from_dict(line_dict['particle'])