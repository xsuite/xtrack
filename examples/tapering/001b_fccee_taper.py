import json
import xtrack as xt

# Build line and tracker
with open('line_no_radiation.json', 'r') as f:
    line = xt.Line.from_dict(json.load(f))

tracker = line.build_tracker()

# Initial twiss (no radiation)
tracker.configure_radiation(model=None)
tw_no_rad = tracker.twiss(method='4d', freeze_longitudinal=True)

# Enable radiation
tracker.configure_radiation(model='mean')

# - Set cavity lags to compensate energy loss
# - Taper magnet strengths
tracker.compensate_radiation_energy_loss()

# Twiss(es) with radiation
tw = tracker.twiss(method='6d', matrix_stability_tol=3.,
                    eneloss_and_damping=True)

import matplotlib.pyplot as plt
plt.close('all')

print(f'Tune error - error_qx: '
      f'{abs(tw.qx - tw_no_rad.qx):.3e}'
      ' error_qy: '
      f'{abs(tw.qy - tw_no_rad.qy):.3e}')

plt.figure(2)

plt.subplot(2,1,1)
plt.plot(tw_no_rad.s, tw.betx/tw_no_rad.betx - 1)
plt.ylabel(r'$\Delta \beta_x / \beta_x$')

plt.subplot(2,1,2)
plt.plot(tw_no_rad.s, tw.bety/tw_no_rad.bety - 1)
plt.ylabel(r'$\Delta \beta_y / \beta_y$')

plt.figure(10)
plt.subplot(2,1,1)
plt.plot(tw_no_rad.s, tw.x, 'g')

plt.subplot(2,1,2)
plt.plot(tw_no_rad.s, tw.y, 'g')

plt.figure(3)
plt.subplot()
plt.plot(tw_no_rad.s, tracker.delta_taper)
plt.plot(tw.s, tw.delta)

plt.show()