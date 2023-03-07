import json
import numpy as np
import xtrack as xt


case_name = 'fcc-ee'
filename = 'line_no_radiation.json'

with open(filename, 'r') as f:
    line = xt.Line.from_dict(json.load(f))

#line['qc1l1.1..1'].knl[0] += 1e-5
#line['qc1l1.1..1'].ksl[0] += 1e-6/5
#line['qc1l1.1..1'].ksl[1] = 1e-4

line.build_tracker(global_xy_limit=20e-2)

for ee in line.elements:
    if ee.__class__.__name__.startswith('Cavity'):
        ee.voltage = 0

# Initial twiss (no radiation)
line.configure_radiation(model=None)
tw_no_rad = line.twiss(method='4d', freeze_longitudinal=True)

# Enable radiation
line.configure_radiation(model='mean')

p = tw_no_rad.particle_on_co.copy()

line.track(p, turn_by_turn_monitor='ONE_TURN_EBE')
mon = line.record_last_track
mon.x[mon.state<1] = np.nan
mon.y[mon.state<1] = np.nan
mon.delta[mon.state<1] = np.nan

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
ax1 = plt.subplot(211)
plt.plot(mon.s.T, mon.x.T, label='x')
plt.plot(mon.s.T, mon.y.T, label='y')
plt.legend(loc='lower left')
plt.ylabel('x, y [m]')
ax2 = plt.subplot(212, sharex=ax1)
plt.plot(mon.s.T, mon.delta.T)
plt.xlabel('s [m]')
plt.ylabel(r'$\delta$')

plt.show()