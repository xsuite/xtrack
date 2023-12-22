import xtrack as xt

d_zeta = 1e-3

collider = xt.Multiline.from_json(
    '../../test_data/hllhc15_collider/collider_00_from_mad.json'
)
collider.build_trackers()


line = collider.lhcb1
line.cycle('ip2', inplace=True)
tw4d = line.twiss(method='4d')

tw_plus = line.twiss(method='4d', start=0, end=len(line) - 1,
                     init=xt.TwissInit(element_name=line.element_names[0],
                                             line=line, zeta=d_zeta))
tw_minus = line.twiss(method='4d', start=0, end=len(line) - 1,
                        init=xt.TwissInit(element_name=line.element_names[0],
                                                line=line, zeta=-d_zeta))
dx_zeta_rf_off_crab_off = (tw_plus.x - tw_minus.x)/(tw_plus.zeta - tw_minus.zeta)

tw_plus_closed = line.twiss(method='4d', zeta0=d_zeta)
tw_minus_closed = line.twiss(method='4d', zeta0=-d_zeta)
cl_dx_zeta_rf_off_crab_off = (tw_plus_closed.x - tw_minus_closed.x)/(tw_plus_closed.zeta - tw_minus_closed.zeta)

line.vars['vrf400'] = 16

tw_plus = line.twiss(method='4d', start=0, end=len(line) - 1,
                     init=xt.TwissInit(element_name=line.element_names[0],
                                             line=line, zeta=d_zeta))
tw_minus = line.twiss(method='4d', start=0, end=len(line) - 1,
                        init=xt.TwissInit(element_name=line.element_names[0],
                                                line=line, zeta=-d_zeta))
dx_zeta_4d_rf_on_crab_off = (tw_plus.x - tw_minus.x)/(tw_plus.zeta - tw_minus.zeta)

tw_plus_closed = line.twiss(method='4d', zeta0=d_zeta)
tw_minus_closed = line.twiss(method='4d', zeta0=-d_zeta)
cl_dx_zeta_4d_rf_on_crab_off = (tw_plus_closed.x - tw_minus_closed.x)/(tw_plus_closed.zeta - tw_minus_closed.zeta)

line.vars['on_crab1'] = -190

tw_plus = line.twiss(method='4d', start=0, end=len(line) - 1,
                        init=xt.TwissInit(element_name=line.element_names[0],
                                                line=line, zeta=d_zeta))
tw_minus = line.twiss(method='4d', start=0, end=len(line) - 1,
                        init=xt.TwissInit(element_name=line.element_names[0],
                                                line=line, zeta=-d_zeta))
dx_zeta_4d_rf_on_crab_on = (tw_plus.x - tw_minus.x)/(tw_plus.zeta - tw_minus.zeta)

tw_plus_closed = line.twiss(method='4d', zeta0=d_zeta)
tw_minus_closed = line.twiss(method='4d', zeta0=-d_zeta)
cl_dx_zeta_4d_rf_on_crab_on = (tw_plus_closed.x - tw_minus_closed.x)/(tw_plus_closed.zeta - tw_minus_closed.zeta)

line.vars['vrf400'] = 0
tw_plus = line.twiss(method='4d', start=0, end=len(line) - 1,
                        init=xt.TwissInit(element_name=line.element_names[0],
                                                line=line, zeta=d_zeta))
tw_minus = line.twiss(method='4d', start=0, end=len(line) - 1,
                        init=xt.TwissInit(element_name=line.element_names[0],
                                                line=line, zeta=-d_zeta))
dx_zeta_rf_off_crab_on = (tw_plus.x - tw_minus.x)/(tw_plus.zeta - tw_minus.zeta)

tw_plus_closed = line.twiss(method='4d', zeta0=d_zeta)
tw_minus_closed = line.twiss(method='4d', zeta0=-d_zeta)
cl_dx_zeta_rf_off_crab_on = (tw_plus_closed.x - tw_minus_closed.x)/(tw_plus_closed.zeta - tw_minus_closed.zeta)


import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.plot(tw_plus.s, dx_zeta_4d_rf_on_crab_on, label='rf on, crab on', color='g')
plt.plot(tw_plus.s, dx_zeta_rf_off_crab_on, label='rf off, crab on', color='m')
plt.plot(tw_plus.s, dx_zeta_4d_rf_on_crab_off, label='rf on, crab off', color='b')
plt.plot(tw_plus.s, dx_zeta_rf_off_crab_off, label='rf off, crab off', color='r')
plt.axvline(x=line.get_s_position('ip4'), color='k', linestyle='--', label='ip4')
plt.legend()
plt.xlabel('s [m]')
plt.ylabel('dx/dzeta')





line.vars['vrf400'] = 16

line.vars['on_crab1'] = 0
tw_crab_off = line.twiss()
line.vars['on_crab1'] = -190
tw_crab_on = line.twiss()

plt.figure(2)
plt.plot(tw_crab_on.s, tw_crab_on.dx_zeta, label='crab on')
plt.plot(tw_crab_off.s, tw_crab_off.dx_zeta, label='crab off')
plt.legend()
plt.xlabel('s [m]')
plt.ylabel('dx/dzeta')


plt.figure(10)
plt.plot(tw_plus.s, cl_dx_zeta_4d_rf_on_crab_on, label='rf on, crab on', color='g')
plt.plot(tw_plus.s, cl_dx_zeta_rf_off_crab_on, label='rf off, crab on', color='m')
plt.plot(tw_plus.s, cl_dx_zeta_4d_rf_on_crab_off, label='rf on, crab off', color='b')
plt.plot(tw_plus.s, cl_dx_zeta_rf_off_crab_off, label='rf off, crab off', color='r')
plt.plot(tw_plus.s, tw_crab_on.dx_zeta, '--', label='6d', color='k', linewidth=2)
plt.legend()
plt.xlabel('s [m]')
plt.ylabel('dx/dzeta')

plt.show()


