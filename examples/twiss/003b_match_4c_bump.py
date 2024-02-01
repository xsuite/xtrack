import json

import numpy as np
import xtrack as xt

with open('../../test_data/hllhc14_no_errors_with_coupling_knobs/line_b1.json',
            'r') as fid:
    dct = json.load(fid)
line = xt.Line.from_dict(dct)

line.build_tracker()

tw_before = line.twiss()

line.match(
    # Portion of the beam line to be modified and initial conditions
    start='mq.33l8.b1',
    end='mq.23l8.b1',
    init=tw_before.get_twiss_init(at_element='mq.33l8.b1'),
    # Dipole corrector strengths to be varied
    vary=[
        xt.Vary(name='acbv30.l8b1', step=1e-10),
        xt.Vary(name='acbv28.l8b1', step=1e-10),
        xt.Vary(name='acbv26.l8b1', step=1e-10),
        xt.Vary(name='acbv24.l8b1', step=1e-10),
    ],
    targets=[
        # I want the vertical orbit to be at 3 mm at mq.28l8.b1 with zero angle
        xt.Target('y', at='mb.b28l8.b1', value=3e-3, tol=1e-4, scale=1),
        xt.Target('py', at='mb.b28l8.b1', value=0, tol=1e-6, scale=1000),
        # I want the bump to be closed
        xt.Target('y', at='mq.23l8.b1', value=tw_before['y', 'mq.23l8.b1'],
                  tol=1e-6, scale=1),
        xt.Target('py', at='mq.23l8.b1', value=tw_before['py', 'mq.23l8.b1'],
                  tol=1e-7, scale=1000),
    ]
)

#!end-doc-part

tw = line.twiss()

import matplotlib.pyplot as plt
plt.close('all')
fig = plt.figure(1, figsize=(6.4*1.2, 4.8*0.8))
ax = fig.add_subplot(111)
ax.plot(tw_before.s, tw_before.y*1000, label='y')
ax.plot(tw.s, tw.y*1000, label='y')
ax.axvline(x=line.get_s_position('mb.b28l8.b1'), color='r', linestyle='--', alpha=0.5)
ax.axvline(x=line.get_s_position('mcbv.30l8.b1'), color='k', linestyle='--', alpha=0.5)
ax.axvline(x=line.get_s_position('mcbv.28l8.b1'), color='k', linestyle='--', alpha=0.5)
ax.axvline(x=line.get_s_position('mcbv.26l8.b1'), color='k', linestyle='--', alpha=0.5)
ax.axvline(x=line.get_s_position('mcbv.24l8.b1'), color='k', linestyle='--', alpha=0.5)
ax.axvline(x=line.get_s_position('mq.33l8.b1'), color='g', linestyle='--', alpha=0.5)
ax.axvline(x=line.get_s_position('mq.23l8.b1'), color='g', linestyle='--', alpha=0.5)
ax.set_xlim(line.get_s_position('mq.33l8.b1') - 10,
            line.get_s_position('mq.23l8.b1') + 10)
ax.set_xlabel('s [m]')
ax.set_ylabel('y [mm]')
ax.set_ylim(-0.5, 10)
plt.subplots_adjust(bottom=.152, top=.9, left=.1, right=.95)
plt.show()

assert np.isclose(tw['y', 'mb.b28l8.b1'], 3e-3, atol=1e-4)
assert np.isclose(tw['py', 'mb.b28l8.b1'], 0, atol=1e-6)
assert np.isclose(tw['y', 'mq.23l8.b1'], tw_before['y', 'mq.23l8.b1'], atol=1e-6)
assert np.isclose(tw['py', 'mq.23l8.b1'], tw_before['py', 'mq.23l8.b1'], atol=1e-7)
assert np.isclose(tw['y', 'mq.33l8.b1'], tw_before['y', 'mq.33l8.b1'], atol=1e-6)
assert np.isclose(tw['py', 'mq.33l8.b1'], tw_before['py', 'mq.33l8.b1'], atol=1e-7)

# There is a bit of leakage in the horizontal plane (due to feed-down from sextupoles)
assert np.isclose(tw['x', 'mb.b28l8.b1'], tw_before['x', 'mb.b28l8.b1'], atol=100e-6)
assert np.isclose(tw['px', 'mb.b28l8.b1'], tw_before['px', 'mb.b28l8.b1'], atol=100e-7)
assert np.isclose(tw['x', 'mq.23l8.b1'], tw_before['x', 'mq.23l8.b1'], atol=100e-6)
assert np.isclose(tw['px', 'mq.23l8.b1'], tw_before['px', 'mq.23l8.b1'], atol=100e-7)
assert np.isclose(tw['x', 'mq.33l8.b1'], tw_before['x', 'mq.33l8.b1'], atol=100e-6)
assert np.isclose(tw['px', 'mq.33l8.b1'], tw_before['px', 'mq.33l8.b1'], atol=100e-7)

# Now I match the bump including the horizontal plane

# I start from scratch
line.vars['acbv30.l8b1'] = 0
line.vars['acbv28.l8b1'] = 0
line.vars['acbv26.l8b1'] = 0
line.vars['acbv24.l8b1'] = 0

line.match(
    start='mq.33l8.b1',
    end='mq.23l8.b1',
    init=tw_before.get_twiss_init(at_element='mq.33l8.b1'),
    vary=[
        xt.Vary(name='acbv30.l8b1', step=1e-10),
        xt.Vary(name='acbv28.l8b1', step=1e-10),
        xt.Vary(name='acbv26.l8b1', step=1e-10),
        xt.Vary(name='acbv24.l8b1', step=1e-10),
        xt.Vary(name='acbh27.l8b1', step=1e-10),
        xt.Vary(name='acbh25.l8b1', step=1e-10),
    ],
    targets=[
        # I want the vertical orbit to be at 3 mm at mq.28l8.b1 with zero angle
        xt.Target('y', at='mb.b28l8.b1', value=3e-3, tol=1e-4, scale=1),
        xt.Target('py', at='mb.b28l8.b1', value=0, tol=1e-6, scale=1000),
        # I want the bump to be closed
        xt.Target('y', at='mq.23l8.b1', value=tw_before['y', 'mq.23l8.b1'],
                  tol=1e-6, scale=1),
        xt.Target('py', at='mq.23l8.b1', value=tw_before['py', 'mq.23l8.b1'],
                  tol=1e-7, scale=1000),
        xt.Target('x', at='mq.23l8.b1', value=tw_before['x', 'mq.23l8.b1'],
                  tol=1e-6, scale=1),
        xt.Target('px', at='mq.23l8.b1', value=tw_before['px', 'mq.23l8.b1'],
                  tol=1e-7, scale=1000),
    ]
)

tw = line.twiss()
assert np.isclose(tw['y', 'mb.b28l8.b1'], 3e-3, atol=1e-4)
assert np.isclose(tw['py', 'mb.b28l8.b1'], 0, atol=1e-6)
assert np.isclose(tw['y', 'mq.23l8.b1'], tw_before['y', 'mq.23l8.b1'], atol=1e-6)
assert np.isclose(tw['py', 'mq.23l8.b1'], tw_before['py', 'mq.23l8.b1'], atol=1e-7)
assert np.isclose(tw['y', 'mq.33l8.b1'], tw_before['y', 'mq.33l8.b1'], atol=1e-6)
assert np.isclose(tw['py', 'mq.33l8.b1'], tw_before['py', 'mq.33l8.b1'], atol=1e-7)

# There is a bit of leakage in the horizontal plane (due to feed-down from sextupoles)
assert np.isclose(tw['x', 'mb.b28l8.b1'], tw_before['x', 'mb.b28l8.b1'], atol=50e-6)
assert np.isclose(tw['px', 'mb.b28l8.b1'], tw_before['px', 'mb.b28l8.b1'], atol=2e-6)
assert np.isclose(tw['x', 'mq.23l8.b1'], tw_before['x', 'mq.23l8.b1'], atol=1e-6)
assert np.isclose(tw['px', 'mq.23l8.b1'], tw_before['px', 'mq.23l8.b1'], atol=1e-7)
assert np.isclose(tw['x', 'mq.33l8.b1'], tw_before['x', 'mq.33l8.b1'], atol=1e-6)
assert np.isclose(tw['px', 'mq.33l8.b1'], tw_before['px', 'mq.33l8.b1'], atol=1e-7)