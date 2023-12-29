import xtrack as xt

# Load a line and build a tracker
collider = xt.Multiline.from_json(
    '../../test_data/hllhc15_thick/hllhc15_collider_thick.json')
collider.build_trackers()

tw0 = collider.twiss(method='4d')

twb1 = collider.lhcb1.twiss(start='e.ds.l5.b1', end='s.ds.r5.b1', init=tw0.lhcb1)
twb2 = collider.lhcb2.twiss(start='e.ds.l5.b2', end='s.ds.r5.b2', init=tw0.lhcb2)

opt = collider.match(
    vary=xt.VaryList([
        'acbxv1.r5', 'acbxv1.l5', # <-- common elements
        'acbyvs4.l5b1', 'acbrdv4.r5b1', 'acbcv5.l5b1', # <-- b1
        'acbyvs4.l5b2', 'acbrdv4.r5b2', 'acbcv5.r5b2', # <-- b2
        ],
        step=1e-10, limits=[-1e-3, 1e-3]),
    targets = [
        twb1.target(y=0, py=10e-6, at='ip5'),
        twb2.target(y=0, py=-10e-6, at='ip5'),
        twb1.target(y=0, py=0, at=xt.END),
        twb2.target(y=0, py=0, at=xt.END),
        twb1.target('py', xt.GreaterThan(9.9e-6), at='ip5'), # <-- inequality
    ])
opt.target_status()

# prints:
#
# Target status:
# id state tag tol_met      residue  current_val         target_val description
#  0 ON           True -5.42101e-20 -5.42101e-20                  0 line=lhcb1, ('y', 'ip5'), val=0, tol=1e- ...
#  1 ON           True -1.99849e-17        9e-06 GreaterThan(9e-06) line=lhcb1, ('py', 'ip5'), val=GreaterTh ...
#  2 ON           True            0        9e-06  LessThan(1.1e-05) line=lhcb1, ('py', 'ip5'), val=LessThan( ...
#  3 ON           True -4.67562e-19 -4.67562e-19                  0 line=lhcb2, ('y', 'ip5'), val=0, tol=1e- ...
#  4 ON           True  1.03338e-19  1.03338e-19                  0 callable, val=0, tol=1e-10, weight=1
#  5 ON           True  3.64674e-18  3.64674e-18                  0 line=lhcb1, ('y', 's.ds.r5.b1'), val=0,  ...
#  6 ON           True  1.68179e-19  1.68179e-19                  0 line=lhcb1, ('py', 's.ds.r5.b1'), val=0, ...
#  7 ON           True  2.05694e-18  2.05694e-18                  0 line=lhcb2, ('y', 's.ds.r5.b2'), val=0,  ...
#  8 ON           True -1.21224e-19 -1.21224e-19                  0 line=lhcb2, ('py', 's.ds.r5.b2'), val=0, ...

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


plt.show()
