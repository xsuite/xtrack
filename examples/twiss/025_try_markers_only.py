import numpy as np
import xtrack as xt

line = xt.Line.from_json(
    '../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')
line.build_tracker()
line.twiss_default['method'] = '4d'

tt = line.get_table()
mask_twiss = np.zeros(len(tt) + 1, dtype=bool)
mask_twiss[:-1] = tt.element_type == 'Marker'

tw_init_ip5 = line.twiss().get_twiss_init('ip5')

tw = line.twiss(ele_start='ip5', ele_stop='s.ds.r5.b1', twiss_init=tw_init_ip5)

line.tracker.mask_twiss = mask_twiss
tw2 = line.twiss(ele_start='ip5', ele_stop='s.ds.r5.b1', twiss_init=tw_init_ip5)
