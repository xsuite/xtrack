import numpy as np
import xtrack as xt

d_zeta = 1e-3

collider = xt.Multiline.from_json(
    '../../test_data/hllhc15_collider/collider_00_from_mad.json'
)
collider.build_trackers()

collider.vars['vrf400'] = 16
collider.vars['on_crab1'] = -190
collider.vars['on_crab5'] = -190

line = collider.lhcb1

tw6d_rf_on = line.twiss()
tw4d_rf_on = line.twiss(method='4d')

collider.vars['vrf400'] = 0
tw4d_rf_off = line.twiss(method='4d')

collider.vars['vrf400'] = 16
collider.vars['on_crab1'] = 0
collider.vars['on_crab5'] = 0

line = collider.lhcb1

tw6d_rf_on_crab_off = line.twiss()
tw4d_rf_on_crab_off = line.twiss(method='4d')

assert np.allclose(tw4d_rf_on['delta'], 0, rtol=0, atol=1e-12)
assert np.allclose(tw4d_rf_off['delta'], 0, rtol=0, atol=1e-12)
assert np.allclose(tw4d_rf_on_crab_off['delta'], 0, rtol=0, atol=1e-12)

assert np.isclose(tw6d_rf_on['dx_zeta', 'ip1'], -190e-6, rtol=1e-4, atol=0)
assert np.isclose(tw6d_rf_on['dy_zeta', 'ip5'], -190e-6, rtol=1e-4, atol=0)
assert np.isclose(tw4d_rf_on['dx_zeta', 'ip1'], -190e-6, rtol=1e-4, atol=0)
assert np.isclose(tw4d_rf_on['dy_zeta', 'ip5'], -190e-6, rtol=1e-4, atol=0)
assert np.isclose(tw4d_rf_off['dx_zeta', 'ip1'], -190e-6, rtol=1e-4, atol=0)
assert np.isclose(tw4d_rf_off['dy_zeta', 'ip5'], -190e-6, rtol=1e-4, atol=0)

assert np.allclose(tw6d_rf_on_crab_off['dx_zeta'], 0, rtol=0, atol=1e-8)
assert np.allclose(tw6d_rf_on_crab_off['dy_zeta'], 0, rtol=0, atol=1e-8)
assert np.allclose(tw4d_rf_on_crab_off['dx_zeta'], 0, rtol=0, atol=1e-8)
assert np.allclose(tw4d_rf_on_crab_off['dy_zeta'], 0, rtol=0, atol=1e-8)

assert np.allclose(tw6d_rf_on['dx_zeta'], tw4d_rf_on['dx_zeta'], rtol=0, atol=1e-7)
assert np.allclose(tw6d_rf_on['dy_zeta'], tw4d_rf_on['dy_zeta'], rtol=0, atol=1e-7)
assert np.allclose(tw6d_rf_on['dx_zeta'], tw4d_rf_off['dx_zeta'], rtol=0, atol=1e-7)
assert np.allclose(tw6d_rf_on['dy_zeta'], tw4d_rf_off['dy_zeta'], rtol=0, atol=1e-7)

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
sp1 = plt.subplot(1,1,1)
plt.plot(tw6d_rf_on.s, tw6d_rf_on.dx_zeta, label='6D - RF on')
plt.plot(tw4d_rf_on.s, tw4d_rf_on.dx_zeta, '--', label='4D - RF on')
plt.plot(tw4d_rf_off.s, tw4d_rf_off.dx_zeta, '-.', label='4D - RF off')
plt.legend()
plt.xlabel('s [m]')
plt.ylabel('dx/dzeta')

plt.figure(2)
sp2 = plt.subplot(1,1,1, sharex=sp1, sharey=sp1)
plt.plot(tw6d_rf_on_crab_off.s, tw6d_rf_on_crab_off.dx_zeta, label='6D - RF on, crab off')
plt.plot(tw4d_rf_on_crab_off.s, tw4d_rf_on_crab_off.dx_zeta, '--', label='4D - RF on, crab off')
plt.legend()
plt.xlabel('s [m]')
plt.ylabel('dx/dzeta')
plt.show()



