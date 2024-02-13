import xtrack as xt

# Load a line and build a tracker
collider = xt.Multiline.from_json(
    '../../test_data/hllhc15_thick/hllhc15_collider_thick.json')
collider.build_trackers()

tw0 = collider.twiss(method='4d')

twb1 = collider.lhcb1.twiss(start='e.ds.l5.b1', end='s.ds.r5.b1', init=tw0.lhcb1)
twb2 = collider.lhcb2.twiss(start='e.ds.l5.b2', end='s.ds.r5.b2', init=tw0.lhcb2)
vars = collider.vars
line_b1 = collider.lhcb1

opt = collider.match(
    solve=False,
    vary=xt.VaryList([
        'acbxv1.r5', 'acbxv1.l5', # <-- common elements
        'acbyvs4.l5b1', 'acbrdv4.r5b1', 'acbcv5.l5b1', 'acbcv6.r5b1', # <-- b1
        'acbyvs4.l5b2', 'acbrdv4.r5b2', 'acbcv5.r5b2', 'acbcv6.l5b2'  # <-- b2
        ],
        step=1e-10, limits=[-1e-3, 1e-3]),
    targets = [
        # Targets from b1 twiss
        twb1.target(y=0, py=10e-6, at='ip5'),
        twb1.target(y=0, py=0, at=xt.END),
        # Targets from b2 twiss
        twb2.target(y=0, py=-10e-6, at='ip5'),
        twb2.target(['y', 'py'], at=xt.END), # <-- preserve
        # Targets from vars
        vars.target('acbxv1.l5', xt.LessThan(1e-3)),
        vars.target('acbxv1.l5', xt.GreaterThan(1e-6)),
        vars.target(lambda vv: vv['acbxv1.l5'] + vv['acbxv1.r5'], xt.LessThan(1e-9)),
        # Targets from line
        line_b1.target(lambda ll: ll['mcbrdv.4r5.b1'].ksl[0], xt.GreaterThan(1e-6)),
        line_b1.target(lambda ll: ll['mcbxfbv.a2r5'].ksl[0] + ll['mcbxfbv.a2l5'].ksl[0],
                                xt.LessThan(1e-9)),
    ])
opt.solve()
opt.target_status()

# prints:
#
# Target status:
# id state tag tol_met      residue  current_val         target_val description
#  0 ON           True  6.95245e-17  6.95245e-17                  0 line=lhcb1, ('y', 'ip5'), val=0, tol=1e- ...
#  1 ON           True  -9.7917e-19        1e-05              1e-05 line=lhcb1, ('py', 'ip5'), val=1e-05, to ...
#  2 ON           True -3.83958e-16 -3.83958e-16                  0 line=lhcb1, ('y', 's.ds.r5.b1'), val=0,  ...
#  3 ON           True -1.73842e-17 -1.73842e-17                  0 line=lhcb1, ('py', 's.ds.r5.b1'), val=0, ...
#  4 ON           True -3.81775e-17 -3.81775e-17                  0 line=lhcb2, ('y', 'ip5'), val=0, tol=1e- ...
#  5 ON           True  -1.5128e-18       -1e-05             -1e-05 line=lhcb2, ('py', 'ip5'), val=-1e-05, t ...
#  6 ON           True -3.66729e-17 -3.66729e-17                0.0 line=lhcb2, ('y', 's.ds.r5.b2'), val=0,  ...
#  7 ON           True  1.35054e-18  1.35054e-18               -0.0 line=lhcb2, ('py', 's.ds.r5.b2'), val=-0 ...
#  8 ON           True            0        1e-06    LessThan(0.001) 'acbxv1.l5', val=LessThan(0.001), tol=1e ...
#  9 ON           True -8.47033e-22        1e-06 GreaterThan(1e-06) 'acbxv1.l5', val=GreaterThan(1e-06), tol ...
# 10 ON           True            0        1e-09    LessThan(1e-09) callable, val=LessThan(1e-09), tol=1e-10 ...
# 11 ON           True            0  1.02889e-06 GreaterThan(1e-06) line=lhcb1, callable, val=GreaterThan(1e ...
# 12 ON           True            0        1e-09    LessThan(1e-09) line=lhcb1, callable, val=LessThan(1e-09 ...

#!end-doc-part

import matplotlib.pyplot as plt

tw = collider.twiss()

plt.close('all')
fig = plt.figure(1, figsize=(6.4*1.3, 4.8))
plt.plot(tw.lhcb1.s - tw.lhcb1['s', 'ip5'], tw.lhcb1.y*1000, label='b1', color='blue')
plt.plot(tw.lhcb2.s - tw.lhcb2['s', 'ip5'], tw.lhcb2.y*1000, label='b2', color='red')
plt.xlabel('s [m]')
plt.ylabel('y [mm]')
plt.legend()

for nn in ['mcbxfbv.a2r5', 'mcbxfbv.a2l5', 'ip5',
           'mcbyv.a4l5.b1', 'mcbrdv.4r5.b1', 'mcbcv.5l5.b1',
           'mcbyv.4l5.b2', 'mcbrdv.4r5.b2', 'mcbcv.5r5.b2']:
    if nn.endswith('b2'):
        lname = 'lhcb2'
        ha = 'right'
        color = 'red'
    elif nn.endswith('b1'):
        lname = 'lhcb1'
        ha = 'left'
        color = 'blue'
    elif nn.startswith('mcbx'):
        lname = 'lhcb1'
        ha = 'right'
        color = 'green'
    else:
        lname = 'lhcb1'
        ha = 'left'
        color = 'black'
    plt.axvline(x=tw[lname]['s', nn] - tw[lname]['s', 'ip5'], color=color, linestyle='--', alpha=0.5)
    plt.text(tw[lname]['s', nn] - tw[lname]['s', 'ip5'], -1.1, nn, rotation=90,
            horizontalalignment=ha, verticalalignment='bottom', color=color)

plt.ylim(-1.15, 0.6)
plt.xlim(-250, 250)

import numpy as np
assert np.isclose(tw.lhcb1['y', 'ip5'], 0, rtol=0, atol=1e-9)
assert np.isclose(tw.lhcb1['py', 'ip5'], 10e-6, rtol=0, atol=1e-10)
assert np.isclose(tw.lhcb1['y', 's.ds.r5.b1'], 0, rtol=0, atol=1e-9)
assert np.isclose(tw.lhcb1['py', 's.ds.r5.b1'], 0, rtol=0, atol=1e-9)
assert np.isclose(tw.lhcb2['y', 'ip5'], 0, rtol=0, atol=1e-9)
assert np.isclose(tw.lhcb2['py', 'ip5'], -10e-6, rtol=0, atol=1e-10)
assert np.isclose(tw.lhcb2['y', 's.ds.r5.b2'], 0, rtol=0, atol=1e-9)
assert np.isclose(tw.lhcb2['py', 's.ds.r5.b2'], 0, rtol=0, atol=1e-9)

assert len(opt.targets) == 13
for ii in range(4):
    assert opt.targets[ii].action is opt.targets[0].action
    assert isinstance(opt.targets[ii].action, xt.match.ActionTwiss)
    assert opt.targets[ii].action.line.name == 'lhcb1'

for ii in range(4, 8):
    assert opt.targets[ii].action is opt.targets[4].action
    assert isinstance(opt.targets[ii].action, xt.match.ActionTwiss)
    assert opt.targets[ii].action.line.name == 'lhcb2'

assert isinstance(opt.targets[8].action, xt.line.ActionVars)
assert isinstance(opt.targets[9].action, xt.line.ActionVars)
assert isinstance(opt.targets[10].action, xt.line.ActionVars)
assert isinstance(opt.targets[11].action, xt.line.ActionLine)
assert isinstance(opt.targets[12].action, xt.line.ActionLine)

opt.tag('solution')
opt.reload(0)

assert np.all(opt.target_status(ret=True).tol_met == np.array(
    [ True, False,  True,  True,  True, False,  True,  True,  True,
      False,  True, False,  True]))
opt.reload(tag='solution')
assert np.all(opt.target_status(ret=True).tol_met)
plt.show()
