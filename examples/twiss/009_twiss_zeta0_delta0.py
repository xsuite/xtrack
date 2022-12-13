import numpy as np
from cpymad.madx import Madx

import xtrack as xt
import xpart as xp

mad = Madx()
mad.call('../../test_data/hllhc15_noerrors_nobb/sequence_with_crabs.madx')
mad.use('lhcb1')
mad.globals.on_crab1 = -190
mad.globals.on_crab5 = -190

line = xt.Line.from_madx_sequence(mad.sequence.lhcb1)
line.particle_ref = xp.Particles(p0c=7000e9, mass0=xp.PROTON_MASS_EV)

tracker = line.build_tracker()

# Measure crabbing angle at IP1 and IP5
z1 = 1e-4
z2 = -1e-4

tw1 = tracker.twiss(zeta0=z1).to_pandas()
tw2 = tracker.twiss(zeta0=z2).to_pandas()

tw1.set_index('name', inplace=True)
tw2.set_index('name', inplace=True)

phi_c_ip1 = ((tw1.loc['ip1', 'x'] - tw2.loc['ip1', 'x'])
         /(tw1.loc['ip1', 'zeta'] - tw2.loc['ip1', 'zeta']))

phi_c_ip5 = ((tw1.loc['ip5', 'y'] - tw2.loc['ip5', 'y'])
            /(tw1.loc['ip5', 'zeta'] - tw2.loc['ip5', 'zeta']))

assert np.isclose(phi_c_ip1, -190e-6, atol=1e-7, rtol=0)
assert np.isclose(phi_c_ip5, -190e-6, atol=1e-7, rtol=0)
