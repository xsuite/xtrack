import xtrack as xt

collider = xt.Multiline.from_json(
    '../../test_data/hllhc15_collider/collider_00_from_mad.json'
)
collider.build_trackers()

line = collider.lhcb1
line.cycle('ip2', inplace=True)

line.vars['vrf400'] = 16

d_zeta = 1e-3
tw_plus = line.twiss(method='4d', ele_start=0, ele_stop=len(line) - 1,
                     twiss_init=xt.TwissInit(element_name=line.element_names[0],
                                             line=line, zeta=d_zeta))
tw_minus = line.twiss(method='4d', ele_start=0, ele_stop=len(line) - 1,
                        twiss_init=xt.TwissInit(element_name=line.element_names[0],
                                                line=line, zeta=-d_zeta))
dx_zeta_4d = (tw_plus.x - tw_minus.x)/(tw_plus.zeta - tw_minus.zeta)

