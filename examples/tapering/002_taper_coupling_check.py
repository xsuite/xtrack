import json
import numpy as np
import xtrack as xt


case_name = 'fcc-ee'
filename = 'line_no_radiation.json'

with open(filename, 'r') as f:
    line = xt.Line.from_dict(json.load(f))

tracker = line.build_tracker()

# Initial twiss (no radiation)
tracker.configure_radiation(model=None)
tw_no_rad = tracker.twiss(method='4d', freeze_longitudinal=True)

# Enable radiation
tracker.configure_radiation(model='mean')
# - Set cavity lags to compensate energy loss
# - Taper magnet strengths
tracker.compensate_radiation_energy_loss(record_iterations=True)

import matplotlib.pyplot as plt
plt.close('all')

tw = tracker.twiss(eneloss_and_damping=True)

p0corr = 1 + tracker.delta_taper

plt.figure(1, figsize=(6.4*1.3, 4.8))

betx_beat = tw.betx*p0corr/tw_no_rad.betx-1
bety_beat = tw.bety*p0corr/tw_no_rad.bety-1
max_betx_beat = np.max(np.abs(betx_beat))
max_bety_beat = np.max(np.abs(bety_beat))
spx = plt.subplot(2,1,1)
plt.title(f'error on Qx: {abs(tw.qx - tw_no_rad.qx):.2e}     '
            r'$(\Delta \beta_x / \beta_x)_{max}$ = '
            f'{max_betx_beat:.2e}')
plt.plot(tw.s, betx_beat)
if 'delta_in_beta' in conf:
    plt.plot(tw.s, -tracker.delta_taper, 'k')
plt.ylabel(r'$\Delta \beta_x / \beta_x$')
plt.ylim(np.max([0.01, 1.1 * max_betx_beat])*np.array([-1, 1]))
plt.xlim([0, tw.s[-1]])

plt.subplot(2,1,2, sharex=spx)
plt.title(f'error on Qy: {abs(tw.qy - tw_no_rad.qy):.2e}     '
            r'$(\Delta \beta_y / \beta_y)_{max}$ = '
            f'{max_bety_beat:.2e}')
plt.plot(tw.s, bety_beat)
if 'delta_in_beta' in conf:
    plt.plot(tw.s, -tracker.delta_taper, 'k')
plt.ylabel(r'$\Delta \beta_y / \beta_y$')
plt.ylim(np.max([0.01, 1.1 * max_bety_beat])*np.array([-1, 1]))
plt.xlabel('s [m]')

plt.subplots_adjust(hspace=0.35, top=.85)

plt.figure(100)
for i_iter, mon in enumerate(tracker._tapering_iterations):
    plt.plot(mon.s.T, mon.delta.T,
                label=f'iter {i_iter} - Ene. loss: {-mon.delta[-1, -1]*100:.2f} %')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.ylim(top=0.01)
    plt.xlim(left=0, right=mon.s[-1, -1])
    plt.xlabel('s [m]')
    plt.ylabel(r'$\delta$')

plt.show()
