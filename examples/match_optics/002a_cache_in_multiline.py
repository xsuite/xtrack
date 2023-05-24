import time

import numpy as np

import xtrack as xt
import xdeps as xd

# xt._print.suppress = True

# Load the line
collider = xt.Multiline.from_json(
    '../../test_data/hllhc15_collider/collider_00_from_mad.json')
collider.build_trackers()

collider.lhcb1.twiss_default['method'] = '4d'
collider.lhcb2.twiss_default['method'] = '4d'
collider.lhcb2.twiss_default['reverse'] = True

collider.vars['on_x2'] = 123
collider.vars.cache_active = True

collider.vars['on_x1']._value
collider.vars['on_x5']._value

vsharing = collider._var_sharing
collider._var_sharing = None

collider.vars['on_x1'] = 234
collider.vars['on_x5'] = 123

assert np.isclose(collider['lhcb1'].twiss()['px', 'ip1'], 234e-6, atol=1e-9, rtol=0)
assert np.isclose(collider['lhcb1'].twiss()['py', 'ip5'], 123e-6, atol=1e-9, rtol=0)

try:
    collider.vars['on_x2'] = 234
except RuntimeError:
    pass
else:
    raise ValueError('Should have raised RuntimeError')

collider._var_sharing = vsharing
collider.vars['on_x2'] = 234

assert np.isclose(collider['lhcb1'].twiss()['py', 'ip2'], 234e-6, atol=1e-9, rtol=0)

