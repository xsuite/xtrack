import json

import numpy as np
import xtrack as xt
import xobjects as xo

with open('../../test_data/hllhc14_no_errors_with_coupling_knobs/line_b1.json',
            'r') as fid:
    dct_b1 = json.load(fid)
input_line = xt.Line.from_dict(dct_b1)

# Load line with knobs on correctors only
from cpymad.madx import Madx
mad = Madx()
mad.call('../../test_data/hllhc14_no_errors_with_coupling_knobs/lhcb1_seq.madx')
mad.use(sequence='lhcb1')
input_line_co_ref = xt.Line.from_madx_sequence(mad.sequence.lhcb1,
    deferred_expressions=True,
    expressions_for_element_types=('kicker', 'hkicker', 'vkicker'))

collider = xt.Multiline(
    lines={'lhcb1': input_line, 'lhcb1_co_ref': input_line_co_ref})
input_line_co_ref.particle_ref = input_line.particle_ref.copy()

# Profit to test the dump and load
collider = xt.Multiline.from_dict(collider.to_dict())
collider.build_trackers()

# Wipe out orbit correction from pymask
for kk in collider._var_sharing.data['var_values']:
    if kk.startswith('corr_acb'):
        collider.vars[kk] = 0

collider.vars['on_x1'] = 0
collider.vars['on_x2'] = 0
collider.vars['on_x5'] = 0
collider.vars['on_x8'] = 0
collider.vars['on_sep1'] = 0
collider.vars['on_sep2'] = 0
collider.vars['on_sep5'] = 0
collider.vars['on_sep8'] = 0
collider.vars['on_lhcb'] = 0
collider.vars['on_alice'] = 0

# Check that in both machines the orbit is flat at the ips
for nn in ['lhcb1', 'lhcb1_co_ref']:
    tw = collider[nn].tracker.twiss(method='4d', zeta0=0, delta0=0)
    for ip in ['ip1', 'ip2', 'ip5', 'ip8']:
        assert np.isclose(tw[ip, 'x'], 0, 1e-10)
        assert np.isclose(tw[ip, 'px'], 0, 1e-10)
        assert np.isclose(tw[ip, 'y'], 0, 1e-10)
        assert np.isclose(tw[ip, 'py'], 0, 1e-10)

# Check that the tune knobs work only on line and not on line_co_ref
tw0 = collider['lhcb1'].twiss(method='4d', zeta0=0, delta0=0)
tw0_co_ref = collider['lhcb1_co_ref'].twiss(method='4d', zeta0=0, delta0=0)
collider['lhcb1'].vars['kqtf.b1'] = 1e-5
collider['lhcb1_co_ref'].vars['kqtf.b1'] = 1e-5 # This should not change anything
tw1 = collider['lhcb1'].twiss(method='4d', zeta0=0, delta0=0)
tw1_co_ref = collider['lhcb1_co_ref'].twiss(method='4d', zeta0=0, delta0=0)
assert tw1.qx != tw0.qx
assert tw1_co_ref.qx == tw0_co_ref.qx

# Add correction term to all dipole correctors
collider.vars['on_corr_co'] = 1
for kk in list(collider.vars._owner.keys()):
    if kk.startswith('acb'):
        collider.vars['corr_co_'+kk] = 0
        collider.vars[kk] += (collider.vars['corr_co_'+kk]
                              * collider.vars['on_corr_co'])

# Set some orbit knobs in both machines and switch on experimental magnets
collider.vars['on_x1'] = 250
collider.vars['on_x2'] = 250
collider.vars['on_x5'] = 250
collider.vars['on_x8'] = 250
collider.vars['on_disp'] = 1
collider.vars['on_lhcb'] = 1
collider.vars['on_alice'] = 1

# Introduce dip kick in all triplets (only in line)
collider['lhcb1']['mqxfb.b2l1..11'].knl[0] = 1e-6
collider['lhcb1']['mqxfb.b2l1..11'].ksl[0] = 1.5e-6
collider['lhcb1']['mqxfb.b2r1..11'].knl[0] = 2e-6
collider['lhcb1']['mqxfb.b2r1..11'].ksl[0] = 1e-6
collider['lhcb1']['mqxb.b2l2..11'].knl[0] = 1e-6
collider['lhcb1']['mqxb.b2l2..11'].ksl[0] = 1.5e-6
collider['lhcb1']['mqxb.b2r2..11'].knl[0] = 2e-6
collider['lhcb1']['mqxb.b2r2..11'].ksl[0] = 1e-6
collider['lhcb1']['mqxfb.b2l5..11'].knl[0] = 1e-6
collider['lhcb1']['mqxfb.b2l5..11'].ksl[0] = 1.5e-6
collider['lhcb1']['mqxfb.b2r5..11'].knl[0] = 2e-6
collider['lhcb1']['mqxfb.b2r5..11'].ksl[0] = 1e-6
collider['lhcb1']['mqxb.b2l8..11'].knl[0] = 1e-6
collider['lhcb1']['mqxb.b2l8..11'].ksl[0] = 1.5e-6
collider['lhcb1']['mqxb.b2r8..11'].knl[0] = 2e-6
collider['lhcb1']['mqxb.b2r8..11'].ksl[0] = 1e-6

# Save the two lines to json
with open('collider.json', 'w') as fid:
    dct = collider.to_dict()
    json.dump(dct, fid, cls=xo.JEncoder)