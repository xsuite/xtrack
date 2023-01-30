import json

import numpy as np
import xtrack as xt
import xobjects as xo

# Load line and line_co_ref from json
with open('line_with_orbit_ref.json', 'r') as fid:
    dct = json.load(fid)
line = xt.Line.from_dict(dct['line'])
line_co_ref = xt.Line.from_dict(dct['line_co_ref'])

tracker = line.build_tracker()
tracker_co_ref = line_co_ref.build_tracker()

# Bind variables in the two lines
from shared_knobs import VarSharing
var_sharing = VarSharing(lines = [line, line_co_ref],
                         names=['lhcb1', '_orbit_ref_lhcb1'])

tw_before = tracker.twiss()

import json
with open('corr_co.json') as ff:
    correction_setup = json.load(ff)

from xtrack.tracker import _temp_knobs

for corr_name, corr in correction_setup.items():
    print('Correcting', corr_name)
    with _temp_knobs(tracker, corr['ref_with_knobs']):
        tw_ref = tracker_co_ref.twiss(method='4d', zeta0=0, delta0=0)
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

assert np.isclose(tw['ip1', 'px'], 250e-6, atol=1e-8)
assert np.isclose(tw['ip1', 'py'], 0, atol=1e-8)

assert np.isclose(tw['ip5', 'px'], 0, atol=1e-8)
assert np.isclose(tw['ip5', 'py'], 250e-6, atol=1e-8)

assert tw['ip2', 'px'] > 1e-7 # effect of the spectrometer tilt
assert tw['ip2', 'py'] > 255e-6 # effect of the spectrometer
assert np.isclose(tw['bpmsw.1r2.b1', 'px'], 0, atol=1e-8)  # external angle
assert np.isclose(tw['bpmsw.1r2.b1', 'py'], 250e-6, atol=1e-8) # external angle

assert tw['ip8', 'px'] > 255e-6 # effect of the spectrometer
assert tw['ip8', 'py'] > 1e-6 # effect of the spectrometer tilt
assert np.isclose(tw['bpmsw.1r8.b1', 'px'], 250e-6, atol=1e-8) # external angle
assert np.isclose(tw['bpmsw.1r8.b1', 'py'], 0, atol=1e-8) # external angle


places_to_check = [
 'e.ds.r8.b1',
 'e.ds.r4.b1',
 's.ds.l2.b1',
 's.ds.l6.b1',
 'e.ds.l1.b1',
 'e.ds.l2.b1',
 'e.ds.l5.b1',
 'e.ds.l8.b1',
 's.ds.r1.b1',
 's.ds.r2.b1',
 's.ds.r5.b1',
 's.ds.r8.b1']

for place in places_to_check:
    assert np.isclose(tw[place, 'x'], tw_ref[place, 'x'], atol=1e-6)
    assert np.isclose(tw[place, 'px'], tw_ref[place, 'px'], atol=1e-8)
    assert np.isclose(tw[place, 'y'], tw_ref[place, 'y'], atol=1e-6)
    assert np.isclose(tw[place, 'py'], tw_ref[place, 'py'], atol=1e-8)


import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
sp1 = plt.subplot(2, 1, 1)
plt.plot(tw_ref.s, tw_ref.x, label='ref')
plt.plot(tw_before.s, tw_before.x, label='before')
plt.plot(tw_before.s, tw.x, label='after')
sp2 = plt.subplot(2, 1, 2, sharex=sp1)
plt.plot(tw_ref.s, tw_ref.y, label='ref')
plt.plot(tw_before.s, tw_before.y, label='before')
plt.plot(tw_before.s, tw.y, label='after')
plt.show()