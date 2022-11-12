import json
import numpy as np
from scipy.constants import c as clight
import xtrack as xt

# Build line and tracker
with open('line_no_radiation.json', 'r') as f:
    line = xt.Line.from_dict(json.load(f))

tracker = line.build_tracker()

# Introduce some closed orbit
line[3].knl[0] += 1e-6
line[3].ksl[0] += 1e-6

# Initial twiss (no radiation)
tracker.configure_radiation(mode=None)
tw_no_rad = tracker.twiss(method='4d', freeze_longitudinal=True)

# Enable radiation
tracker.configure_radiation(mode='mean')

# Set cavity lags to compensate energy loss and taper magnet strengths
tracker.compensate_radiation_energy_loss()

# Twiss(es) with radiation
tw_real_tracking = tracker.twiss(method='6d', matrix_stability_tol=3.,
                    eneloss_and_damping=True)
tw_sympl = tracker.twiss(model_radiation='kick_as_co', method='6d')
tw_preserve_angles = tracker.twiss(
                        model_radiation='preserve_angles',
                        method='6d',
                        matrix_stability_tol=0.5)

import matplotlib.pyplot as plt

print('Non sympltectic tracker:')
print(f'Tune error =  error_qx: {abs(tw_real_tracking.qx - tw_no_rad.qx):.3e} error_qy: {abs(tw_real_tracking.qy - tw_no_rad.qy):.3e}')
print('Sympltectic tracker:')
print(f'Tune error =  error_qx: {abs(tw_sympl.qx - tw_no_rad.qx):.3e} error_qy: {abs(tw_sympl.qy - tw_no_rad.qy):.3e}')
print ('Preserve angles:')
print(f'Tune error =  error_qx: {abs(tw_preserve_angles.qx - tw_no_rad.qx):.3e} error_qy: {abs(tw_preserve_angles.qy - tw_no_rad.qy):.3e}')
plt.figure(2)

plt.subplot(2,1,1)
plt.plot(tw_no_rad.s, tw_sympl.betx/tw_no_rad.betx - 1)
plt.plot(tw_no_rad.s, tw_preserve_angles.betx/tw_no_rad.betx - 1)
#tw.betx *= (1 + delta_beta_corr)
#plt.plot(tw_no_rad.s, tw.betx/tw_no_rad.betx - 1)
plt.ylabel(r'$\Delta \beta_x / \beta_x$')

plt.subplot(2,1,2)
plt.plot(tw_no_rad.s, tw_sympl.bety/tw_no_rad.bety - 1)
plt.plot(tw_no_rad.s, tw_preserve_angles.bety/tw_no_rad.bety - 1)
#tw.bety *= (1 + delta_beta_corr)
#plt.plot(tw_no_rad.s, tw.bety/tw_no_rad.bety - 1)
plt.ylabel(r'$\Delta \beta_y / \beta_y$')

plt.figure(10)
plt.subplot(2,1,1)
plt.plot(tw_no_rad.s, tw_no_rad.x, 'k')
#plt.plot(tw_no_rad.s, tw_real_tracking.x, 'b')
plt.plot(tw_no_rad.s, tw_sympl.x, 'r')
plt.plot(tw_no_rad.s, tw_preserve_angles.x, 'g')

plt.subplot(2,1,2)
plt.plot(tw_no_rad.s, tw_no_rad.y, 'k')
#plt.plot(tw_no_rad.s, tw_real_tracking.y, 'b')
plt.plot(tw_no_rad.s, tw_sympl.y, 'r')
plt.plot(tw_no_rad.s, tw_preserve_angles.y, 'g')



plt.show()