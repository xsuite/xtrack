import xtrack as xt

# Load a line and build a tracker
collider = xt.Multiline.from_json(
    '../../test_data/hllhc15_thick/hllhc15_collider_thick.json')
collider.build_trackers()

tw0 = collider.twiss(method='4d')

opt = collider.match(
    lines=['lhcb1', 'lhcb2'],
    start=['mq.30l8.b1', 'mq.31l8.b2'],
    end=['mq.23l8.b1', 'mq.24l8.b2'],
    init=tw0,
    vary=xt.VaryList(['acbv30.l8b1', 'acbv28.l8b1', 'acbv26.l8b1', 'acbv24.l8b1',
                      'acbv31.l8b2', 'acbv29.l8b2', 'acbv27.l8b2', 'acbv25.l8b2'],
                    step=1e-10, limits=[-1e-3, 1e-3]),
    targets = [
        xt.TargetSet(y=3e-3, py=0, at='mb.b28l8.b1', line='lhcb1'),
        xt.TargetSet(y=-3e-3, py=0, at='mb.b28l8.b2', line='lhcb2'),
        xt.TargetSet(y=0, py=0, at=xt.END, line='lhcb1'),
        xt.TargetSet(y=0, py=0, at=xt.END, line='lhcb2'),
    ])

opt.target_status()

#!end-doc-part
import matplotlib.pyplot as plt
import numpy as np
plt.close('all')

line = collider['lhcb1']

opt.tag(tag='matched')
opt.reload(0)
tw_before = line.twiss(method='4d')
opt.reload(tag='matched')
tw = line.twiss(method='4d')

fig = plt.figure(1, figsize=(6.4*1.2, 4.8*0.8))
ax = fig.add_subplot(111)
ax.plot(tw.s, tw.y*1000, label='y')

for nn in ['mcbv.30l8.b1', 'mcbv.28l8.b1', 'mcbv.26l8.b1', 'mcbv.24l8.b1']:
    ax.axvline(x=line.get_s_position(nn), color='k', linestyle='--', alpha=0.5)
    ax.text(line.get_s_position(nn), 10, nn, rotation=90,
            horizontalalignment='left', verticalalignment='top')

ax.axvline(x=line.get_s_position('mb.b28l8.b1'), color='r', linestyle='--', alpha=0.5)
ax.text(line.get_s_position('mb.b28l8.b1'), 10, 'mb.b28l8.b1', rotation=90,
        horizontalalignment='left', verticalalignment='top')

ax.axvline(x=line.get_s_position('mq.30l8.b1'), color='g', linestyle='--', alpha=0.5)
ax.axvline(x=line.get_s_position('mq.23l8.b1'), color='g', linestyle='--', alpha=0.5)
ax.text(line.get_s_position('mq.30l8.b1'), 10, 'mq.30l8.b1', rotation=90,
        horizontalalignment='right', verticalalignment='top')
ax.text(line.get_s_position('mq.23l8.b1'), 10, 'mq.23l8.b1', rotation=90,
        horizontalalignment='right', verticalalignment='top')

ax.set_xlim(line.get_s_position('mq.30l8.b1') - 10,
        line.get_s_position('mq.23l8.b1') + 10)
ax.set_xlabel('s [m]')
ax.set_ylabel('y [mm]')
ax.set_ylim(-0.5, 10)
plt.subplots_adjust(bottom=.152, top=.9, left=.1, right=.95)

assert np.isclose(tw['y', 'mb.b28l8.b1'], 3e-3, atol=1e-4)
assert np.isclose(tw['py', 'mb.b28l8.b1'], 0, atol=1e-6)
assert np.isclose(tw['y', 'mq.23l8.b1'], tw_before['y', 'mq.23l8.b1'], atol=1e-6)
assert np.isclose(tw['py', 'mq.23l8.b1'], tw_before['py', 'mq.23l8.b1'], atol=1e-7)
assert np.isclose(tw['y', 'mq.33l8.b1'], tw_before['y', 'mq.33l8.b1'], atol=1e-6)
assert np.isclose(tw['py', 'mq.33l8.b1'], tw_before['py', 'mq.33l8.b1'], atol=1e-7)


line = collider['lhcb2']

opt.tag(tag='matched')
opt.reload(0)
tw_before = line.twiss(method='4d')
opt.reload(tag='matched')
tw = line.twiss(method='4d')

fig = plt.figure(2, figsize=(6.4*1.2, 4.8*0.8))
ax = fig.add_subplot(111)
ax.plot(tw.s, tw.y*1000, label='y')

for nn in ['mcbv.31l8.b2', 'mcbv.29l8.b2', 'mcbv.27l8.b2', 'mcbv.25l8.b2']:
    ax.axvline(x=tw['s', nn], color='k', linestyle='--', alpha=0.5)
    ax.text(tw['s', nn], -9.8, nn, rotation=90,
            horizontalalignment='left', verticalalignment='bottom')

ax.axvline(x=tw['s', 'mb.b28l8.b2'], color='r', linestyle='--', alpha=0.5)
ax.text(tw['s', 'mb.b28l8.b2'], -9.8, 'mb.b28l8.b2', rotation=90,
        horizontalalignment='left', verticalalignment='bottom')

ax.axvline(x=tw['s', 'mq.31l8.b2'], color='g', linestyle='--', alpha=0.5)
ax.axvline(x=tw['s', 'mq.24l8.b2'], color='g', linestyle='--', alpha=0.5)
ax.text(tw['s', 'mq.31l8.b2'], -9.8, 'mq.31l8.b2', rotation=90,
        horizontalalignment='right', verticalalignment='bottom')
ax.text(tw['s', 'mq.24l8.b2'], -9.8, 'mq.24l8.b2', rotation=90,
        horizontalalignment='right', verticalalignment='bottom')

ax.set_xlim(tw['s', 'mq.31l8.b2'] - 10, tw['s', 'mq.24l8.b2'] + 10)
ax.set_xlabel('s [m]')
ax.set_ylabel('y [mm]')
ax.set_ylim(-10, 0.5)
plt.subplots_adjust(bottom=.152, top=.9, left=.1, right=.95)

assert np.isclose(tw['y', 'mb.b28l8.b2'], -3e-3, atol=1e-4)
assert np.isclose(tw['py', 'mb.b28l8.b2'], 0, atol=1e-6)
assert np.isclose(tw['y', 'mq.24l8.b2'], tw_before['y', 'mq.23l8.b2'], atol=1e-6)
assert np.isclose(tw['py', 'mq.24l8.b2'], tw_before['py', 'mq.23l8.b2'], atol=1e-7)
assert np.isclose(tw['y', 'mq.33l8.b2'], tw_before['y', 'mq.33l8.b2'], atol=1e-6)
assert np.isclose(tw['py', 'mq.33l8.b2'], tw_before['py', 'mq.33l8.b2'], atol=1e-7)

plt.show()