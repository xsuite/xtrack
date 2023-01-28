import json

import numpy as np
import xtrack as xt
import xobjects as xo

with open('../../test_data/hllhc14_no_errors_with_coupling_knobs/line_b1.json',
            'r') as fid:
    dct_b1 = json.load(fid)
line = xt.Line.from_dict(dct_b1)
tracker = line.build_tracker()

# Load line with knobs on correctors only
from cpymad.madx import Madx
mad = Madx()
mad.call('../../test_data/hllhc14_no_errors_with_coupling_knobs/lhcb1_seq.madx')
mad.use(sequence='lhcb1')
line_co_ref = xt.Line.from_madx_sequence(mad.sequence.lhcb1,
    deferred_expressions=True,
    expressions_for_element_types=('kicker', 'hkicker', 'vkicker'))

tracker_co_ref = line_co_ref.build_tracker()
tracker_co_ref.particle_ref = tracker.particle_ref

# Wipe out orbit correction from pymask
for ll in [line_co_ref, line]:
    for kk in ll._var_management['data']['var_values']:
        if kk.startswith('corr_acb'):
            ll.vars[kk] = 0

# Switch off crossing angles and separations
for ll in [line_co_ref, line]:
    ll.vars['on_x1'] = 0
    ll.vars['on_x2'] = 0
    ll.vars['on_x5'] = 0
    ll.vars['on_x8'] = 0
    ll.vars['on_sep1'] = 0
    ll.vars['on_sep2'] = 0
    ll.vars['on_sep5'] = 0
    ll.vars['on_sep8'] = 0

# Switch off experimental magnets
for ll in [line_co_ref, line]:
    ll.vars['on_alice'] = 0
    ll.vars['on_lhcb'] = 0
    ll.vars['on_sol_alice'] = 0
    ll.vars['on_sol_lhcb'] = 0
    ll.vars['on_sol_atlas'] = 0
    ll.vars['on_sol_cms'] = 0

# Check that in both machines the orbit is flat at the ips
for ll in [line_co_ref, line]:
    tw = ll.tracker.twiss(method='4d', zeta0=0, delta0=0)
    for ip in ['ip1', 'ip2', 'ip5', 'ip8']:
        assert np.isclose(tw[ip, 'x'], 0, 1e-10)
        assert np.isclose(tw[ip, 'px'], 0, 1e-10)
        assert np.isclose(tw[ip, 'y'], 0, 1e-10)
        assert np.isclose(tw[ip, 'py'], 0, 1e-10)

# Check that the tune knobs work only on line and not on line_co_ref
tw0 = tracker.twiss(method='4d', zeta0=0, delta0=0)
tw0_co_ref = tracker_co_ref.twiss(method='4d', zeta0=0, delta0=0)
tracker.vars['kqtf.b1'] = 1e-5
tracker_co_ref.vars['kqtf.b1'] = 1e-5
tw1 = tracker.twiss(method='4d', zeta0=0, delta0=0)
tw1_co_ref = tracker_co_ref.twiss(method='4d', zeta0=0, delta0=0)
assert tw1.qx != tw0.qx
assert tw1_co_ref.qx == tw0_co_ref.qx

# Add correction term to all dipole correctors