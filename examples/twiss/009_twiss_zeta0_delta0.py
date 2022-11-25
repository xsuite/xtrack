import json
import numpy as np

import xtrack as xt
import xpart as xp

with open('../../test_data/hllhc_14/line_and_particle.json', 'r') as fid:
    dct = json.load(fid)

line = xt.Line.from_dict(dct['line'])
line.particle_ref = xp.Particles(**dct['particle'])

tracker = line.build_tracker()

# Measure crabbing angle at IP1
z1 = 1e-2
z2 = -1e-2

tw1 = tracker.twiss(zeta0=z1).to_pandas()
tw2 = tracker.twiss(zeta0=z2).to_pandas()

tw1.set_index('name', inplace=True)
tw2.set_index('name', inplace=True)

phi_c = ((tw1.loc['ip1', 'x'] - tw2.loc['ip1', 'x'])
         /(tw1.loc['ip1', 'zeta'] - tw2.loc['ip1', 'zeta']))

assert np.isclose(phi_c, -190e-6, atol=0, rtol=0.02)
