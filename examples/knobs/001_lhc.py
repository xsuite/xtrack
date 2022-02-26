import xtrack as xt
import xpart as xp
import xobjects as xo

from cpymad.madx import Madx

# Load sequence in MAD-X
mad = Madx()
mad.call('../../test_data/hllhc15_noerrors_nobb/sequence.madx')
mad.use(sequence="lhcb1")

# Build
line = xt.Line.from_madx_sequence(mad.sequence['lhcb1'],
                                  deferred_expressions=True # <--
                                  )
# Define reference particle
line.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1,
                                 gamma0=mad.sequence.lhcb1.beam.gamma)

# Build tracker
tracker = xt.Tracker(line=line)


import json
with open('status.json', 'w') as fid:
    json.dump(line.to_dict(), fid,
    cls=xo.JEncoder)

import numpy as np
line.vars['on_x1'] = 250
assert np.isclose(tracker.twiss(at_elements=['ip1'])['px'][0], 250e-6,
                  atol=1e-6, rtol=0)

line.vars['on_x1'] = -300
assert np.isclose(tracker.twiss(at_elements=['ip1'])['px'][0], -300e-6,
                  atol=1e-6, rtol=0)
