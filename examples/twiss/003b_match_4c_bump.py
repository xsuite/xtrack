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

line.match(
    ele_start='mq.33l8.b1',
    ele_stop='mq.23l8.b1',
    twiss_init=tw_before.get_twiss_init(at_element='mq.33l8.b1'),
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
        xt.Target('y', at='mq.23l8.b1', value=tw_before['mq.23l8.b1', 'y'],
                  tol=1e-6, scale=1),
        xt.Target('py', at='mq.23l8.b1', value=tw_before['mq.23l8.b1', 'py'],
                  tol=1e-7, scale=1000),
    ]
)

tw = line.twiss()

import matplotlib.pyplot as plt
plt.close('all')
fig = plt.figure(1, figsize=(6.4*1.5, 4.8))
ax = fig.add_subplot(111)
ax.plot(tw_before.s, tw_before.y*1000, label='y')
ax.plot(tw.s, tw.y*1000, label='y')
ax.axvline(x=line.get_s_position('mb.b28l8.b1'), color='k')
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
plt.show()

assert np.isclose(tw['mb.b28l8.b1', 'y'], 3e-3, atol=1e-4)
assert np.isclose(tw['mb.b28l8.b1', 'py'], 0, atol=1e-6)
assert np.isclose(tw['mq.23l8.b1', 'y'], tw_before['mq.23l8.b1', 'y'], atol=1e-6)
assert np.isclose(tw['mq.23l8.b1', 'py'], tw_before['mq.23l8.b1', 'py'], atol=1e-7)
assert np.isclose(tw['mq.33l8.b1', 'y'], tw_before['mq.33l8.b1', 'y'], atol=1e-6)
assert np.isclose(tw['mq.33l8.b1', 'py'], tw_before['mq.33l8.b1', 'py'], atol=1e-7)

# There is a bit of leakage in the horizontal plane (due to feed-down from sextupoles)
assert np.isclose(tw['mb.b28l8.b1', 'x'], tw_before['mb.b28l8.b1', 'x'], atol=100e-6)
assert np.isclose(tw['mb.b28l8.b1', 'px'], tw_before['mb.b28l8.b1', 'px'], atol=100e-7)
assert np.isclose(tw['mq.23l8.b1', 'x'], tw_before['mq.23l8.b1', 'x'], atol=100e-6)
assert np.isclose(tw['mq.23l8.b1', 'px'], tw_before['mq.23l8.b1', 'px'], atol=100e-7)
assert np.isclose(tw['mq.33l8.b1', 'x'], tw_before['mq.33l8.b1', 'x'], atol=100e-6)
assert np.isclose(tw['mq.33l8.b1', 'px'], tw_before['mq.33l8.b1', 'px'], atol=100e-7)

# Now I match the bump including the horizontal plane

# I start from scratch
line.vars['acbv30.l8b1'] = 0
line.vars['acbv28.l8b1'] = 0
line.vars['acbv26.l8b1'] = 0
line.vars['acbv24.l8b1'] = 0

line.match(
    ele_start='mq.33l8.b1',
    ele_stop='mq.23l8.b1',
    twiss_init=tw_before.get_twiss_init(at_element='mq.33l8.b1'),
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
        xt.Target('y', at='mq.23l8.b1', value=tw_before['mq.23l8.b1', 'y'],
                  tol=1e-6, scale=1),
        xt.Target('py', at='mq.23l8.b1', value=tw_before['mq.23l8.b1', 'py'],
                  tol=1e-7, scale=1000),
        xt.Target('x', at='mq.23l8.b1', value=tw_before['mq.23l8.b1', 'x'],
                  tol=1e-6, scale=1),
        xt.Target('px', at='mq.23l8.b1', value=tw_before['mq.23l8.b1', 'px'],
                  tol=1e-7, scale=1000),
    ]
)

tw = line.twiss()
assert np.isclose(tw['mb.b28l8.b1', 'y'], 3e-3, atol=1e-4)
assert np.isclose(tw['mb.b28l8.b1', 'py'], 0, atol=1e-6)
assert np.isclose(tw['mq.23l8.b1', 'y'], tw_before['mq.23l8.b1', 'y'], atol=1e-6)
assert np.isclose(tw['mq.23l8.b1', 'py'], tw_before['mq.23l8.b1', 'py'], atol=1e-7)
assert np.isclose(tw['mq.33l8.b1', 'y'], tw_before['mq.33l8.b1', 'y'], atol=1e-6)
assert np.isclose(tw['mq.33l8.b1', 'py'], tw_before['mq.33l8.b1', 'py'], atol=1e-7)

# There is a bit of leakage in the horizontal plane (due to feed-down from sextupoles)
assert np.isclose(tw['mb.b28l8.b1', 'x'], tw_before['mb.b28l8.b1', 'x'], atol=50e-6)
assert np.isclose(tw['mb.b28l8.b1', 'px'], tw_before['mb.b28l8.b1', 'px'], atol=2e-6)
assert np.isclose(tw['mq.23l8.b1', 'x'], tw_before['mq.23l8.b1', 'x'], atol=1e-6)
assert np.isclose(tw['mq.23l8.b1', 'px'], tw_before['mq.23l8.b1', 'px'], atol=1e-7)
assert np.isclose(tw['mq.33l8.b1', 'x'], tw_before['mq.33l8.b1', 'x'], atol=1e-6)
assert np.isclose(tw['mq.33l8.b1', 'px'], tw_before['mq.33l8.b1', 'px'], atol=1e-7)