import json
import numpy as np
import xtrack as xt


with open('../../test_data/clic_dr/line_for_taper.json', 'r') as f:
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

assert np.isclose(tracker.delta_taper[0], 0, rtol=0, atol=1e-10)
assert np.isclose(tracker.delta_taper[-1], 0, rtol=0, atol=1e-10)
assert np.isclose(np.max(tracker.delta_taper), 0.00568948, rtol=1e-4, atol=0)
assert np.isclose(np.min(tracker.delta_taper), -0.00556288, rtol=1e-4, atol=0)

assert np.allclose(tw_real_tracking.delta, tracker.delta_taper, rtol=0, atol=1e-6)
assert np.allclose(tw_sympl.delta, tracker.delta_taper, rtol=0, atol=1e-6)
assert np.allclose(tw_scale_as_co.delta, tracker.delta_taper, rtol=0, atol=1e-6)

assert np.isclose(tw_real_tracking.qx, tw_no_rad.qx, rtol=0, atol=5e-4)
assert np.isclose(tw_sympl.qx, tw_no_rad.qx, rtol=0, atol=5e-4)
assert np.isclose(tw_scale_as_co.qx, tw_no_rad.qx, rtol=0, atol=5e-4)

assert np.isclose(tw_real_tracking.qy, tw_no_rad.qy, rtol=0, atol=5e-4)
assert np.isclose(tw_sympl.qy, tw_no_rad.qy, rtol=0, atol=5e-4)
assert np.isclose(tw_scale_as_co.qy, tw_no_rad.qy, rtol=0, atol=5e-4)

assert np.isclose(tw_real_tracking.dqx, tw_no_rad.dqx, rtol=0, atol=0.2)
assert np.isclose(tw_sympl.dqx, tw_no_rad.dqx, rtol=0, atol=0.2)
assert np.isclose(tw_scale_as_co.dqx, tw_no_rad.dqx, rtol=0, atol=0.2)

assert np.isclose(tw_real_tracking.dqy, tw_no_rad.dqy, rtol=0, atol=0.2)
assert np.isclose(tw_sympl.dqy, tw_no_rad.dqy, rtol=0, atol=0.2)
assert np.isclose(tw_scale_as_co.dqy, tw_no_rad.dqy, rtol=0, atol=0.2)

assert np.allclose(tw_real_tracking.x, tw_no_rad.x, rtol=0, atol=1e-7)
assert np.allclose(tw_sympl.x, tw_no_rad.x, rtol=0, atol=1e-7)
assert np.allclose(tw_scale_as_co.x, tw_no_rad.x, rtol=0, atol=1e-7)

assert np.allclose(tw_real_tracking.y, tw_no_rad.y, rtol=0, atol=1e-7)
assert np.allclose(tw_sympl.y, tw_no_rad.y, rtol=0, atol=1e-7)
assert np.allclose(tw_scale_as_co.y, tw_no_rad.y, rtol=0, atol=1e-7)

assert np.allclose(tw_sympl.betx, tw_no_rad.betx, rtol=0.02, atol=0)
assert np.allclose(tw_scale_as_co.betx, tw_no_rad.betx, rtol=0.003, atol=0)

assert np.allclose(tw_sympl.bety, tw_no_rad.bety, rtol=0.04, atol=0)
assert np.allclose(tw_scale_as_co.bety, tw_no_rad.bety, rtol=0.003, atol=0)

assert np.allclose(tw_sympl.dx, tw_no_rad.dx, rtol=0.00, atol=0.1e-3)
assert np.allclose(tw_scale_as_co.dx, tw_no_rad.dx, rtol=0.00, atol=0.1e-3)

assert np.allclose(tw_sympl.dy, tw_no_rad.dy, rtol=0.00, atol=0.1e-3)
assert np.allclose(tw_scale_as_co.dy, tw_no_rad.dy, rtol=0.00, atol=0.1e-3)

eneloss_real_tracking = tw_real_tracking.eneloss_turn
assert np.isclose(line['rf'].voltage*np.sin(line['rf'].lag/180*np.pi), eneloss_real_tracking/4, rtol=1e-5)
assert np.isclose(line['rf1'].voltage*np.sin(line['rf1'].lag/180*np.pi), eneloss_real_tracking/4, rtol=1e-5)
assert np.isclose(line['rf2a'].voltage*np.sin(line['rf2a'].lag/180*np.pi), eneloss_real_tracking/4*0.6, rtol=1e-5)
assert np.isclose(line['rf2b'].voltage*np.sin(line['rf2b'].lag/180*np.pi), eneloss_real_tracking/4*0.4, rtol=1e-5)
assert np.isclose(line['rf3'].voltage*np.sin(line['rf3'].lag/180*np.pi), eneloss_real_tracking/4, rtol=1e-5)
plt.show()