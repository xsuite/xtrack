import pickle

import numpy as np

import xtrack as xt
import xpart as xp

from cpymad.madx import Madx

# xt._print.suppress = True

# Load the line
line = xt.Line.from_json(
    '../../test_data/hllhc15_noerrors_nobb/line_w_knobs_and_particle.json')
line.particle_ref = xp.Particles(p0c=7e12, mass=xp.PROTON_MASS_EV)
line.build_tracker()

# Try to pickle line.vars

ss = pickle.dumps(line.vars)
vars = pickle.loads(ss)


lnss = pickle.dumps(line)
ln = pickle.loads(lnss)

# Check that expressions work on old and new line
line.vars['on_x1'] = 234
ln.vars['on_x1'] = 123
assert np.isclose(line.twiss(method='4d')['px', 'ip1'], 234e-6, atol=1e-9, rtol=0)
assert np.isclose(ln.twiss(method='4d')['px', 'ip1'], 123e-6, atol=1e-9, rtol=0)

ln.vars['on_x1'] = 321
assert np.isclose(line.twiss(method='4d')['px', 'ip1'], 234e-6, atol=1e-9, rtol=0)
assert np.isclose(ln.twiss(method='4d')['px', 'ip1'], 321e-6, atol=1e-9, rtol=0)

line.vars['on_x1'] = 213
assert np.isclose(line.twiss(method='4d')['px', 'ip1'], 213e-6, atol=1e-9, rtol=0)
assert np.isclose(ln.twiss(method='4d')['px', 'ip1'], 321e-6, atol=1e-9, rtol=0)

line.discard_tracker()

collider = xt.Multiline(lines={'lhcb1': line})
collider.build_trackers()

colliderss = pickle.dumps(collider)
coll = pickle.loads(colliderss)

collider.vars['on_x1'] = 234
coll.vars['on_x1'] = 123
assert np.isclose(collider['lhcb1'].twiss(method='4d')['px', 'ip1'], 234e-6, atol=1e-9, rtol=0)
assert np.isclose(coll['lhcb1'].twiss(method='4d')['px', 'ip1'], 123e-6, atol=1e-9, rtol=0)

coll.vars['on_x1'] = 321
assert np.isclose(collider['lhcb1'].twiss(method='4d')['px', 'ip1'], 234e-6, atol=1e-9, rtol=0)
assert np.isclose(coll['lhcb1'].twiss(method='4d')['px', 'ip1'], 321e-6, atol=1e-9, rtol=0)

collider.vars['on_x1'] = 213
assert np.isclose(collider['lhcb1'].twiss(method='4d')['px', 'ip1'], 213e-6, atol=1e-9, rtol=0)
assert np.isclose(coll['lhcb1'].twiss(method='4d')['px', 'ip1'], 321e-6, atol=1e-9, rtol=0)

try:
    lnss2 = pickle.dumps(line)
except RuntimeError: # Cannot pickle a line within a multiline
    pass
else:
    raise RuntimeError('Should have raised RuntimeError')


