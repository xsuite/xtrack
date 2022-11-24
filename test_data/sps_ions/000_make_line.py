import json
import numpy as np

import xtrack as xt
import xpart as xp
import xobjects as xo
from cpymad.madx import Madx

mad = Madx()
mad.call('SPS_2021_Pb_ions_thin_test.seq')
mad.use('sps')
mad.twiss()

twmad_4d = mad.table.twiss.dframe()
summad_4d = mad.table.summ.dframe()

V_RF = 1.7e6 # V (control room definition, energy gain per charge)

# I switch on one cavity
charge = mad.sequence.sps.beam.charge
mad.sequence.sps.elements['actcse.31632'].volt = V_RF/1e6 * charge
mad.sequence.sps.elements['actcse.31632'].lag = 0
mad.sequence.sps.elements['actcse.31632'].freq = 200.

twmad_6d = mad.table.twiss.dframe()
summad_6d = mad.table.summ.dframe()

mad.emit()
qs_mad = mad.table.emitsumm.qs[0]

# Make xsuite line and tracker
line = xt.Line.from_madx_sequence(mad.sequence.sps, deferred_expressions=True)
line.particle_ref = xp.Particles(mass0=mad.sequence.sps.beam.mass*1e9,
                                 q0=mad.sequence.sps.beam.charge,
                                 gamma0=mad.sequence.sps.beam.gamma)
tracker = line.build_tracker()

tw = tracker.twiss()

assert np.isclose(tw.qs, qs_mad, atol=1e-6)
assert np.isclose(tw.qx, summad_4d.q1, atol=1e-5)
assert np.isclose(tw.qy, summad_4d.q2, atol=1e-5)
assert np.isclose(tw.dqx, summad_6d.dq1, atol=0.5)
assert np.isclose(tw.dqy, summad_6d.dq2, atol=0.5)

with open('line_and_particle.json', 'w') as f:
    json.dump(line.to_dict(), f, cls=xo.JEncoder, indent=4)

