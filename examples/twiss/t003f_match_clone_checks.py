import json

import numpy as np
import xtrack as xt
import xobjects as xo

with open('../../test_data/hllhc14_no_errors_with_coupling_knobs/line_b1.json',
            'r') as fid:
    dct = json.load(fid)
line = xt.Line.from_dict(dct)

line.build_tracker()

tw_before = line.twiss()

GreaterThan = xt.GreaterThan
LessThan = xt.LessThan

tw0 = line.twiss()
opt = line.match(
    name='bump',
    solve=False,
    solver='jacobian',
    # Portion of the beam line to be modified and initial conditions
    start='mq.33l8.b1',
    end='mq.17l8.b1',
    init=tw0, init_at=xt.START,
    # Dipole corrector strengths to be varied
    vary=[
        xt.Vary(name='acbv32.l8b1', step=1e-10, weight=0.7),
        xt.Vary(name='acbv28.l8b1', step=1e-10, weight=0.3),
        xt.Vary(name='acbv26.l8b1', step=1e-10),
        xt.Vary(name='acbv24.l8b1', step=1e-10),
        xt.Vary(name='acbv22.l8b1', step=1e-10, limits=[-38e-6, 38e-6], weight=1000),
        xt.Vary(name='acbv18.l8b1', step=1e-10),
    ],
    targets=[
        xt.Target('py', at='mb.b26l8.b1', value=0, tol=1e-6, weight=1e3),
        xt.Target('y', at='mb.b26l8.b1', value=3e-3, tol=1e-4),
        xt.Target('y', at='mq.17l8.b1', value=tw0, tol=1e-6),
        xt.Target('py', at='mq.17l8.b1', value=tw0, tol=1e-7, weight=1e3),
    ]
)

# Check target_mismatch
assert opt.name == 'bump'
ts = opt.target_status(ret=True)
assert len(ts) == 4
assert np.all(ts.tol_met == np.array([True, False, True, True]))
tm = opt.target_mismatch(ret=True)
assert len(tm) == 1
assert tm.id[0] == 1

opt.solve()

# I want to limit the negative excursion ot the bump
opt2 = opt.clone(name='limit',
    add_targets=[
        xt.Target('y', GreaterThan(-2e-3), at='mq.30l8.b1', tol=1e-6),
        xt.Target('y', GreaterThan(-1e-3), at='mq.30l8.b1', tol=1e-6)])
opt2.solve()

assert opt2.name == 'limit'
assert len(opt2.targets) == 6
tm = opt2.target_mismatch(ret=True)
assert(len(tm) == 0)

tw = line.twiss()

assert np.isclose(tw['y', 'mq.33l8.b1'], 0, atol=1e-6, rtol=0)
assert np.isclose(tw['y', 'mq.17l8.b1'], 0, atol=1e-6, rtol=0)
assert np.isclose(tw['py', 'mq.17l8.b1'], 0, atol=1e-8, rtol=0)
assert np.isclose(tw['py', 'mq.33l8.b1'], 0, atol=1e-6, rtol=0)

assert np.isclose(tw['y', 'mb.b26l8.b1'], 3e-3, atol=1e-6, rtol=0)
assert np.isclose(tw['py', 'mb.b26l8.b1'], 0, atol=1e-8, rtol=0)

assert np.isclose(tw['y', 'mq.30l8.b1'], -1e-3, atol=1e-6, rtol=0)
assert np.isclose(line.vars['acbv22.l8b1']._value, 38e-6, atol=0, rtol=0.02)

# Test variable in inequality
line['myvar'] = -5e-3
opt3 = opt2.clone(name='ineq',
    add_targets=[
        xt.Target('y', GreaterThan(line.ref['myvar']), at='mq.30l8.b1', tol=1e-6)])

assert len(opt3.target_mismatch(ret=True)) == 0
assert opt3.target_status(ret=True).residue[-1] == 0

line['myvar'] = -0.5e-3
assert len(opt3.target_mismatch(ret=True)) == 1
xo.assert_allclose(opt3.target_status(ret=True).residue[-1], -0.5e-3,
                   atol=1e-5, rtol=0)

opt3.solve()
assert len(opt3.target_mismatch(ret=True)) == 0
