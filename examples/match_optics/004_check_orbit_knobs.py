import time

import numpy as np

import xtrack as xt
import lhc_match as lm

default_tol = {None: 1e-8, 'betx': 1e-6, 'bety': 1e-6} # to have no rematching w.r.t. madx

collider = xt.Multiline.from_json('collider_02_changed_ip15_phase.json')
collider.build_trackers()

# -----------------------------------------------------------------------------

# Check higher level knobs

collider.vars['on_x2'] = 34
tw = collider.twiss()
collider.vars['on_x2'] = 0

assert np.isclose(tw.lhcb1['py', 'ip2'], 34e-6, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb2['py', 'ip2'], -34e-6, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb1['y', 'ip2'], 0, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb2['y', 'ip2'], 0, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb1['px', 'ip2'], 0, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb2['px', 'ip2'], 0, atol=1e-9, rtol=0)

collider.vars['on_x8'] = 35
tw = collider.twiss()
collider.vars['on_x8'] = 0

assert np.isclose(tw.lhcb1['px', 'ip8'], 35e-6, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb2['px', 'ip8'], -35e-6, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb1['x', 'ip8'], 0, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb2['x', 'ip8'], 0, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb1['py', 'ip8'], 0, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb2['py', 'ip8'], 0, atol=1e-9, rtol=0)

collider.vars['on_sep2'] = 0.5
tw = collider.twiss()
collider.vars['on_sep2'] = 0

assert np.isclose(tw.lhcb1['x', 'ip2'], -0.5e-3, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb2['x', 'ip2'], 0.5e-3, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb1['px', 'ip2'], 0, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb2['px', 'ip2'], 0, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb1['y', 'ip2'], 0, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb2['y', 'ip2'], 0, atol=1e-9, rtol=0)

collider.vars['on_sep8'] = 0.6
tw = collider.twiss()
collider.vars['on_sep8'] = 0

assert np.isclose(tw.lhcb1['y', 'ip8'], 0.6e-3, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb2['y', 'ip8'], -0.6e-3, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb1['py', 'ip8'], 0, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb2['py', 'ip8'], 0, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb1['x', 'ip8'], 0, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb2['x', 'ip8'], 0, atol=1e-9, rtol=0)

# Check lower level knobs (disconnects higher level knobs)

collider.vars['on_o2v'] = 0.3
tw = collider.twiss()
collider.vars['on_o2v'] = 0

assert np.isclose(tw.lhcb1['y', 'ip2'], 0.3e-3, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb2['y', 'ip2'], 0.3e-3, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb1['py', 'ip2'], 0., atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb2['py', 'ip2'], 0., atol=1e-9, rtol=0)

collider.vars['on_o2h'] = 0.4
tw = collider.twiss()
collider.vars['on_o2h'] = 0

assert np.isclose(tw.lhcb1['x', 'ip2'], 0.4e-3, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb2['x', 'ip2'], 0.4e-3, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb1['px', 'ip2'], 0., atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb2['px', 'ip2'], 0., atol=1e-9, rtol=0)

collider.vars['on_o8v'] = 0.5
tw = collider.twiss()
collider.vars['on_o8v'] = 0

assert np.isclose(tw.lhcb1['y', 'ip8'], 0.5e-3, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb2['y', 'ip8'], 0.5e-3, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb1['py', 'ip8'], 0., atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb2['py', 'ip8'], 0., atol=1e-9, rtol=0)

collider.vars['on_o8h'] = 0.6
tw = collider.twiss()
collider.vars['on_o8h'] = 0

assert np.isclose(tw.lhcb1['x', 'ip8'], 0.6e-3, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb2['x', 'ip8'], 0.6e-3, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb1['px', 'ip8'], 0., atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb2['px', 'ip8'], 0., atol=1e-9, rtol=0)

collider.vars['on_a2h'] = 20
tw = collider.twiss()
collider.vars['on_a2h'] = 0

assert np.isclose(tw.lhcb1['x', 'ip2'], 0, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb2['x', 'ip2'], 0, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb1['px', 'ip2'], 20e-6, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb2['px', 'ip2'], 20e-6, atol=1e-9, rtol=0)

collider.vars['on_a2v'] = 15
tw = collider.twiss()
collider.vars['on_a2v'] = 0

assert np.isclose(tw.lhcb1['y', 'ip2'], 0, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb2['y', 'ip2'], 0, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb1['py', 'ip2'], 15e-6, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb2['py', 'ip2'], 15e-6, atol=1e-9, rtol=0)

collider.vars['on_a8h'] = 20
tw = collider.twiss()
collider.vars['on_a8h'] = 0

assert np.isclose(tw.lhcb1['x', 'ip8'], 0, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb2['x', 'ip8'], 0, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb1['px', 'ip8'], 20e-6, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb2['px', 'ip8'], 20e-6, atol=1e-9, rtol=0)

collider.vars['on_a8v'] = 50
tw = collider.twiss()
collider.vars['on_a8v'] = 0

assert np.isclose(tw.lhcb1['y', 'ip8'], 0, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb2['y', 'ip8'], 0, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb1['py', 'ip8'], 50e-6, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb2['py', 'ip8'], 50e-6, atol=1e-9, rtol=0)

collider.vars['on_x2v'] = 100
tw = collider.twiss()
collider.vars['on_x2v'] = 0

assert np.isclose(tw.lhcb1['y', 'ip2'], 0, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb2['y', 'ip2'], 0, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb1['py', 'ip2'], 100e-6, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb2['py', 'ip2'], -100e-6, atol=1e-9, rtol=0)

collider.vars['on_x2h'] = 120
tw = collider.twiss()
collider.vars['on_x2h'] = 0

assert np.isclose(tw.lhcb1['x', 'ip2'], 0, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb2['x', 'ip2'], 0, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb1['px', 'ip2'], 120e-6, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb2['px', 'ip2'], -120e-6, atol=1e-9, rtol=0)


collider.vars['on_x8h'] = 100
tw = collider.twiss()
collider.vars['on_x8h'] = 0

assert np.isclose(tw.lhcb1['x', 'ip8'], 0, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb2['x', 'ip8'], 0, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb1['px', 'ip8'], 100e-6, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb2['px', 'ip8'], -100e-6, atol=1e-9, rtol=0)

collider.vars['on_x8v'] = 120
tw = collider.twiss()
collider.vars['on_x8v'] = 0

assert np.isclose(tw.lhcb1['y', 'ip8'], 0, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb2['y', 'ip8'], 0, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb1['py', 'ip8'], 120e-6, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb2['py', 'ip8'], -120e-6, atol=1e-9, rtol=0)

collider.vars['on_sep2h'] = 1.6
tw = collider.twiss()
collider.vars['on_sep2h'] = 0

assert np.isclose(tw.lhcb1['x', 'ip2'], 1.6e-3, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb2['x', 'ip2'], -1.6e-3, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb1['px', 'ip2'], 0, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb2['px', 'ip2'], 0, atol=1e-9, rtol=0)

collider.vars['on_sep2v'] = 1.7
tw = collider.twiss()
collider.vars['on_sep2v'] = 0

assert np.isclose(tw.lhcb1['y', 'ip2'], 1.7e-3, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb2['y', 'ip2'], -1.7e-3, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb1['py', 'ip2'], 0, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb2['py', 'ip2'], 0, atol=1e-9, rtol=0)

collider.vars['on_sep8h'] = 1.5
tw = collider.twiss()
collider.vars['on_sep8h'] = 0

assert np.isclose(tw.lhcb1['x', 'ip8'], 1.5e-3, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb2['x', 'ip8'], -1.5e-3, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb1['px', 'ip8'], 0, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb2['px', 'ip8'], 0, atol=1e-9, rtol=0)

collider.vars['on_sep8v'] = 1.7
tw = collider.twiss()
collider.vars['on_sep8v'] = 0

assert np.isclose(tw.lhcb1['y', 'ip8'], 1.7e-3, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb2['y', 'ip8'], -1.7e-3, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb1['py', 'ip8'], 0, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb2['py', 'ip8'], 0, atol=1e-9, rtol=0)

# Both knobs together
collider.vars['on_x8h'] = 120
collider.vars['on_sep8h'] = 1.7
tw = collider.twiss()
collider.vars['on_x8h'] = 0
collider.vars['on_sep8h'] = 0

assert np.isclose(tw.lhcb1['x', 'ip8'], 1.7e-3, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb2['x', 'ip8'], -1.7e-3, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb1['px', 'ip8'], 120e-6, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb2['px', 'ip8'], -120e-6, atol=1e-9, rtol=0)
