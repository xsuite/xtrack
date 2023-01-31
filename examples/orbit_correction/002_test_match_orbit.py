import json

import numpy as np
import xtrack as xt
import xobjects as xo

from multiline import Multiline

# Load line and line_co_ref from json
with open('line_with_orbit_ref.json', 'r') as fid:
    dct = json.load(fid)
line = xt.Line.from_dict(dct['line'])
line_co_ref = xt.Line.from_dict(dct['line_co_ref'])

collider = Multiline(lines={'lhcb1': line, 'lhcb1_co_ref': line_co_ref})

collider.build_trackers()

line = collider.lhcb1

tw_before = line.twiss()

import json
with open('corr_co.json') as ff:
    correction_config = json.load(ff)

line.correct_closed_orbit(tracker_co_ref=line_co_ref,
                             correction_config=correction_config)

tw = line.twiss()

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
    assert np.isclose(tw[place, 'x'], 0, atol=1e-6)
    assert np.isclose(tw[place, 'px'], 0, atol=1e-8)
    assert np.isclose(tw[place, 'y'], 0, atol=1e-6)
    assert np.isclose(tw[place, 'py'], 0, atol=1e-8)

with xt.tracker._temp_knobs(line, dict(on_corr_co=0, on_disp=0)):
    tw_ref = line_co_ref.twiss()

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
plt.legend()
plt.show()