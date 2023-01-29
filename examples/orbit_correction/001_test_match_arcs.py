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
for ll in [line_co_ref, line]:
    for kk in list(ll._var_management['data']['var_values'].keys()):
        if kk.startswith('acb'):
            ll.vars['corr_co_'+kk] = 0
            ll.vars[kk] += ll.vars['corr_co_'+kk]

# Set some orbit knobs in both machines
for ll in [line_co_ref, line]:
    ll.vars['on_x1'] = 250
    ll.vars['on_x2'] = 250
    ll.vars['on_x5'] = 250
    ll.vars['on_x8'] = 250
    ll.vars['on_disp'] = 1

# Introduce dip kick in all triplets (only in line)
line['mqxfb.b2l1..11'].knl[0] = 1e-6
line['mqxfb.b2l1..11'].ksl[0] = 1.5e-6
line['mqxfb.b2r1..11'].knl[0] = 2e-6
line['mqxfb.b2r1..11'].ksl[0] = 1e-6

# line['mqxb.b2l2..11'].knl[0] = 1e-6
# line['mqxb.b2l2..11'].ksl[0] = 1.5e-6
# line['mqxb.b2r2..11'].knl[0] = 2e-6
# line['mqxb.b2r2..11'].ksl[0] = 1e-6

# line['mqxfb.b2l5..11'].knl[0] = 1e-6
# line['mqxfb.b2l5..11'].ksl[0] = 1.5e-6
# line['mqxfb.b2r5..11'].knl[0] = 2e-6
# line['mqxfb.b2r5..11'].ksl[0] = 1e-6

# line['mqxb.b2l8..11'].knl[0] = 1e-6
# line['mqxb.b2l8..11'].ksl[0] = 1.5e-6
# line['mqxb.b2r8..11'].knl[0] = 2e-6
# line['mqxb.b2r8..11'].ksl[0] = 1e-6

tw_before = tracker.twiss()

# Use line ref to get an orbit reference
line_co_ref.vars['on_disp'] = 0
tw_ref = line_co_ref.tracker.twiss(method='4d', zeta0=0, delta0=0)

correction_setup = {
    'IR1 left': dict(
        start='e.ds.r8.b1',
        end='e.ds.l1.b1',
        vary=(
            'corr_co_acbh14.l1b1',
            'corr_co_acbh12.l1b1',
            'corr_co_acbv15.l1b1',
            'corr_co_acbv13.l1b1',
            ),
        targets=('e.ds.l1.b1',),
    ),
    'IR1 right': dict(
        start='s.ds.r1.b1',
        end='s.ds.l2.b1',
        vary=(
            'corr_co_acbh13.r1b1',
            'corr_co_acbh15.r1b1',
            'corr_co_acbv12.r1b1',
            'corr_co_acbv14.r1b1',
            ),
        targets=('s.ds.l2.b1',),
    ),
    'IR5 left': dict(
        start='e.ds.r4.b1',
        end='e.ds.l5.b1',
        vary=(
            'corr_co_acbh14.l5b1',
            'corr_co_acbh12.l5b1',
            'corr_co_acbv15.l5b1',
            'corr_co_acbv13.l5b1',
            ),
        targets=('e.ds.l5.b1',),
    ),
    'IR5 right': dict(
        start='s.ds.r5.b1',
        end='s.ds.l6.b1',
        vary=(
            'corr_co_acbh13.r5b1',
            'corr_co_acbh15.r5b1',
            'corr_co_acbv12.r5b1',
            'corr_co_acbv14.r5b1',
            ),
        targets=('s.ds.l6.b1',),
    ),
    'IP1': dict(
        start='e.ds.l1.b1',
        end='s.ds.r1.b1',
        vary=(
            'corr_co_acbch6.l1b1',
            'corr_co_acbcv5.l1b1',
            'corr_co_acbch5.r1b1',
            'corr_co_acbcv6.r1b1',
            'corr_co_acbyhs4.l1b1',
            'corr_co_acbyhs4.r1b1',
            'corr_co_acbyvs4.l1b1',
            'corr_co_acbyvs4.r1b1',
        ),
        targets=('ip1', 's.ds.r1.b1'),
    ),
    'IP2': dict(
        start='e.ds.l2.b1',
        end='s.ds.r2.b1',
        vary=(
            'corr_co_acbyhs5.l2b1',
            'corr_co_acbchs5.r2b1',
            'corr_co_acbyvs5.l2b1',
            'corr_co_acbcvs5.r2b1',
            'corr_co_acbyhs4.l2b1',
            'corr_co_acbyhs4.r2b1',
            'corr_co_acbyvs4.l2b1',
            'corr_co_acbyvs4.r2b1',
        ),
        targets=('ip2', 's.ds.r2.b1'),
    ),
    'IP5': dict(
        start='e.ds.l5.b1',
        end='s.ds.r5.b1',
        vary=(
            'corr_acbch6.l5b1',
            'corr_acbcv5.l5b1',
            'corr_acbch5.r5b1',
            'corr_acbcv6.r5b1',
            'corr_acbyhs4.l5b1',
            'corr_acbyhs4.r5b1',
            'corr_acbyvs4.l5b1',
            'corr_acbyvs4.r5b1',
        ),
        targets=('ip5', 's.ds.r5.b1'),
    ),
    'IP8': dict(
        start='e.ds.l8.b1',
        end='s.ds.r8.b1',
        vary=(
            'corr_acbch5.l8b1',
            'corr_acbyhs4.l8b1',
            'corr_acbyhs4.r8b1',
            'corr_acbyhs5.r8b1',
            'corr_acbcvs5.l8b1',
            'corr_acbyvs4.l8b1',
            'corr_acbyvs4.r8b1',
            'corr_acbyvs5.r8b1',
        ),
        targets=('ip8', 's.ds.r8.b1'),
    ),
}

for corr_name, corr in correction_setup.items():
    print('Correcting', corr_name)
    vary = [xt.Vary(vv, step=1e-9, limits=[-5e-6, 5e-6]) for vv in corr['vary']]
    targets = []
    for tt in corr['targets']:
        assert isinstance(tt, str), 'For now only strings are supported for targets'
        for kk in ['x', 'px', 'y', 'py']:
            targets.append(xt.Target(kk, at=tt, value=tw_ref[tt, kk], tol=1e-9))

    tracker.match(
        vary=vary,
        targets=targets,
        twiss_init=xt.OrbitOnly(
            x=tw_ref[corr['start'], 'x'],
            px=tw_ref[corr['start'], 'px'],
            y=tw_ref[corr['start'], 'y'],
            py=tw_ref[corr['start'], 'py'],
            zeta=tw_ref[corr['start'], 'zeta'],
            delta=tw_ref[corr['start'], 'delta'],
        ),
        ele_start=corr['start'], ele_stop=corr['end'])

tw = tracker.twiss()

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
sp1 = plt.subplot(2, 1, 1)
plt.plot(tw_before.s, tw_before.x, label='before')
plt.plot(tw_before.s, tw.x, label='ref')
sp2 = plt.subplot(2, 1, 2, sharex=sp1)
plt.plot(tw_before.s, tw_before.y, label='before')
plt.plot(tw_before.s, tw.y, label='ref')
plt.show()