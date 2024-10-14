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
        # I want the vertical orbit to be at 3 mm at mq.28l8.b1 with zero angle
        xt.Target('y', at='mb.b26l8.b1', value=3e-3, tol=1e-4),
        xt.Target('py', at='mb.b26l8.b1', value=0, tol=1e-6, weight=1e3),
        # I want the bump to be closed
        xt.Target('y', at='mq.17l8.b1', value=tw0, tol=1e-6),
        xt.Target('py', at='mq.17l8.b1', value=tw0, tol=1e-7, weight=1e3),
        # I want to limit the negative excursion ot the bump
        xt.Target('y', -2e-3, at='mq.30l8.b1', tol=1e-6),
        xt.Target('y', GreaterThan(-1e-3), at='mq.30l8.b1', tol=1e-6),
        # xt.Target(lambda tw: -tw['y', 'mq.30l8.b1'], LessThan(1e-3))
    ]
)
opt.targets[-1].active = False
opt.solve()
opt.targets[-1].active = True
opt.targets[-2].active = False
opt.solve()

opt.solve()

#!end-doc-part

tw = line.twiss()

import matplotlib.pyplot as plt
plt.close('all')
fig = plt.figure(1, figsize=(6.4*1.2, 4.8*0.8))
ax = fig.add_subplot(111)
ax.plot(tw_before.s, tw_before.y*1000, label='y')
ax.plot(tw.s, tw.y*1000, label='y')
# Target
ax.axvline(x=line.get_s_position('mb.b26l8.b1'), color='r', linestyle='--', alpha=0.5)
# Correctors
ax.axvline(x=line.get_s_position('mcbv.32l8.b1'), color='k', linestyle='--', alpha=0.5)
ax.axvline(x=line.get_s_position('mcbv.28l8.b1'), color='k', linestyle='--', alpha=0.5)
ax.axvline(x=line.get_s_position('mcbv.26l8.b1'), color='k', linestyle='--', alpha=0.5)
ax.axvline(x=line.get_s_position('mcbv.24l8.b1'), color='k', linestyle='--', alpha=0.5)
ax.axvline(x=line.get_s_position('mcbv.22l8.b1'), color='k', linestyle='--', alpha=0.5)
ax.axvline(x=line.get_s_position('mcbv.18l8.b1'), color='k', linestyle='--', alpha=0.5)
# Boundaries
ax.axvline(x=line.get_s_position('mq.33l8.b1'), color='g', linestyle='--', alpha=0.5)
ax.axvline(x=line.get_s_position('mq.17l8.b1'), color='g', linestyle='--', alpha=0.5)
ax.axhline(y=-1, color='b', linestyle='--', alpha=0.5)
ax.set_xlim(line.get_s_position('mq.33l8.b1') - 10,
            line.get_s_position('mq.17l8.b1') + 10)
ax.set_xlabel('s [m]')
ax.set_ylabel('y [mm]')
ax.set_ylim(-10, 10)
plt.subplots_adjust(bottom=.152, top=.9, left=.1, right=.95)
plt.show()

assert np.isclose(tw['y', 'mq.33l8.b1'], 0, atol=1e-6, rtol=0)
assert np.isclose(tw['y', 'mq.17l8.b1'], 0, atol=1e-6, rtol=0)
assert np.isclose(tw['py', 'mq.17l8.b1'], 0, atol=1e-8, rtol=0)
assert np.isclose(tw['py', 'mq.33l8.b1'], 0, atol=1e-6, rtol=0)

assert np.isclose(tw['y', 'mb.b26l8.b1'], 3e-3, atol=1e-6, rtol=0)
assert np.isclose(tw['py', 'mb.b26l8.b1'], 0, atol=1e-8, rtol=0)

assert np.isclose(tw['y', 'mq.30l8.b1'], -1e-3, atol=1e-6, rtol=0)
assert np.isclose(line.vars['acbv22.l8b1']._value, 38e-6, atol=0, rtol=0.02)
