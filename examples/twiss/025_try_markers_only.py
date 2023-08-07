import numpy as np
import xtrack as xt

collider = xt.Multiline.from_json(
    '../../test_data/hllhc15_thick/hllhc15_collider_thick.json')
collider.build_trackers()
collider.lhcb1.twiss_default['method'] = '4d'
collider.lhcb2.twiss_default['method'] = '4d'
collider.lhcb2.twiss_default['reverse'] = True

line = collider.lhcb1

tw_init_ip5 = line.twiss().get_twiss_init('s.ds.l5.b1')

tw = line.twiss(ele_start='s.ds.l5.b1', ele_stop='e.ds.r5.b1', twiss_init=tw_init_ip5)
tw2 = line.twiss(ele_start='s.ds.l5.b1', ele_stop='e.ds.r5.b1', twiss_init=tw_init_ip5)

tw_mk = line.twiss(ele_start='s.ds.l5.b1', ele_stop='e.ds.r5.b1', twiss_init=tw_init_ip5,
                   only_markers=True)
tw2_mk = line.twiss(ele_start='s.ds.l5.b1', ele_stop='e.ds.r5.b1', twiss_init=tw_init_ip5,
                    only_markers=True)

line = collider.lhcb2

tw_init_ip5 = line.twiss().get_twiss_init('s.ds.l5.b2')

import pdb; pdb.set_trace()
tw = line.twiss(ele_start='s.ds.l5.b2', ele_stop='e.ds.r5.b2', twiss_init=tw_init_ip5)
tw2 = line.twiss(ele_start='s.ds.l5.b2', ele_stop='e.ds.r5.b2', twiss_init=tw_init_ip5)

tw_mk = line.twiss(ele_start='s.ds.l5.b2', ele_stop='e.ds.r5.b2', twiss_init=tw_init_ip5,
                   only_markers=True)
tw2_mk = line.twiss(ele_start='s.ds.l5.b2', ele_stop='e.ds.r5.b2', twiss_init=tw_init_ip5,
                    only_markers=True)


assert tw_mk['s', 'e.ds.r5.b2'] == tw['s', 'e.ds.r5.b2']
