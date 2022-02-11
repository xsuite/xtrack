import json
import sys
import math
import numpy as np

import xtrack as xt
import xpart as xp
import xobjects as xo
import xdeps as xd
from xdeps.madxutils import MadxEval

from cpymad.madx import Madx
mad = Madx(command_log="mad_final.log")
mad.call('../../test_data/hllhc15_noerrors_nobb/sequence.madx')
mad.use(sequence="lhcb1")
mad.globals['vrf400'] = 16
mad.globals['lagrf400.b1'] = 0.5
mad.twiss()

line = xt.Line.from_madx_sequence(
        mad.sequence['lhcb1'], apply_madx_errors=False,
        deferred_expressions=True)
line.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1,
                        gamma0=mad.sequence.lhcb1.beam.gamma)

tracker = xt.Tracker(line=line)

line.vars['on_x1'] = 250
assert np.isclose(tracker.twiss(at_elements=['ip1'])['px'][0], 250e-6,
                  atol=1e-6, rtol=0)

line.vars['on_x1'] = -300
assert np.isclose(tracker.twiss(at_elements=['ip1'])['px'][0], -300e-6,
                  atol=1e-6, rtol=0)

with open('status.json', 'w') as fid:
    json.dump(line.to_dict(), fid,
    cls=xo.JEncoder)