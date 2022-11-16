import json
import xtrack as xt

# Build line and tracker
with open('line_no_radiation.json', 'r') as f:
    line = xt.Line.from_dict(json.load(f))

tracker = line.build_tracker()

# Introduce some closed orbit
line['qc1l1.1..1'].knl[0] += 1e-6
line['qc1l1.1..1'].ksl[0] += 1e-6

# Initial twiss (no radiation)
tracker.configure_radiation(model=None)
tw_no_rad = tracker.twiss(method='4d', freeze_longitudinal=True)

# Enable radiation
tracker.configure_radiation(model='mean')

# - Set cavity lags to compensate energy loss
# - Taper magnet strengths
tracker.compensate_radiation_energy_loss()

# Twiss(es) with radiation
tw_real_tracking = tracker.twiss(method='6d', matrix_stability_tol=3.,
                    eneloss_and_damping=True)
tw_sympl = tracker.twiss(radiation_method='kick_as_co', method='6d')
tw_scale_as_co = tracker.twiss(
                        radiation_method='scale_as_co',
                        method='6d',
                        matrix_stability_tol=0.5)


import matplotlib.pyplot as plt
plt.close('all')

print('Non sympltectic tracker:')
print(f'Tune error =  error_qx: {abs(tw_real_tracking.qx - tw_no_rad.qx):.3e} error_qy: {abs(tw_real_tracking.qy - tw_no_rad.qy):.3e}')
print('Sympltectic tracker:')
print(f'Tune error =  error_qx: {abs(tw_sympl.qx - tw_no_rad.qx):.3e} error_qy: {abs(tw_sympl.qy - tw_no_rad.qy):.3e}')
print ('Preserve angles:')
print(f'Tune error =  error_qx: {abs(tw_scale_as_co.qx - tw_no_rad.qx):.3e} error_qy: {abs(tw_scale_as_co.qy - tw_no_rad.qy):.3e}')
plt.figure(2)

plt.subplot(2,1,1)
plt.plot(tw_no_rad.s, tw_sympl.betx/tw_no_rad.betx - 1)
plt.plot(tw_no_rad.s, tw_scale_as_co.betx/tw_no_rad.betx - 1)
#tw.betx *= (1 + delta_beta_corr)
#plt.plot(tw_no_rad.s, tw.betx/tw_no_rad.betx - 1)
plt.ylabel(r'$\Delta \beta_x / \beta_x$')

plt.subplot(2,1,2)
plt.plot(tw_no_rad.s, tw_sympl.bety/tw_no_rad.bety - 1)
plt.plot(tw_no_rad.s, tw_scale_as_co.bety/tw_no_rad.bety - 1)
#tw.bety *= (1 + delta_beta_corr)
#plt.plot(tw_no_rad.s, tw.bety/tw_no_rad.bety - 1)
plt.ylabel(r'$\Delta \beta_y / \beta_y$')

plt.figure(10)
plt.subplot(2,1,1)
plt.plot(tw_no_rad.s, tw_no_rad.x, 'k')
#plt.plot(tw_no_rad.s, tw_real_tracking.x, 'b')
plt.plot(tw_no_rad.s, tw_sympl.x, 'r')
plt.plot(tw_no_rad.s, tw_scale_as_co.x, 'g')

plt.subplot(2,1,2)
plt.plot(tw_no_rad.s, tw_no_rad.y, 'k')
#plt.plot(tw_no_rad.s, tw_real_tracking.y, 'b')
plt.plot(tw_no_rad.s, tw_sympl.y, 'r')
plt.plot(tw_no_rad.s, tw_scale_as_co.y, 'g')

plt.figure(3)
plt.subplot()
plt.plot(tw_no_rad.s, tracker.delta_taper)
plt.plot(tw_real_tracking.s, tw_real_tracking.delta)

plt.show()