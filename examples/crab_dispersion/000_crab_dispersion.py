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



