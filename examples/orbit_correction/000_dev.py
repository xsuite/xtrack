import json
import numpy as np

import xtrack as xt
import xpart as xp

fname = '../../test_data/hllhc15_noerrors_nobb/line_w_knobs_and_particle.json'

with open(fname) as fid:
    line = xt.Line.from_dict(json.load(fid)['line'])
line.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1, p0c=7e12)
tracker = line.build_tracker()

tw = tracker.twiss()

# Introduce small dipolar errors
line_df = line.to_pandas()
# select mumltipoles
multipole_df = line_df[line_df['element_type'] == 'Multipole']

for ee in multipole_df.element:
    if len(ee.knl)>1 and ee.knl[1] != 0:
        ee.knl[0] += np.random.normal(0, 1e-5)*ee.knl[1]
        ee.ksl[0] += np.random.normal(0, 1e-5)*ee.knl[1]

tw = tracker.twiss()


