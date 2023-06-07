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

tw_plus_closed = line.twiss(
    method='4d', zeta0=d_zeta, delta0=0, freeze_longitudinal=True)
tw_minus_closed = line.twiss(
    method='4d', zeta0=-d_zeta, delta0=0, freeze_longitudinal=True)
cl_dx_zeta_4d_rf_on_crab_off = (tw_plus_closed.x - tw_minus_closed.x)/(tw_plus_closed.zeta - tw_minus_closed.zeta)

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.plot(tw_plus_closed.s, cl_dx_zeta_4d_rf_on_crab_off, label='rf on, crab on', color='g')

plt.show()