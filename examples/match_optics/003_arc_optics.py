import numpy as np
import xtrack as xt

import lhc_match as lm

collider = xt.Multiline.from_json('hllhc.json')
collider.build_trackers()
collider.vars.load_madx_optics_file(
    "../../test_data/hllhc15_thick/opt_round_150_1500.madx")

arc_periodic_solution =lm.get_arc_periodic_solution(collider)

twpresq = collider.lhcb1.twiss(
    twiss_init=xt.TwissInit(
        element_name='ip1', betx=0.5, bety=0.5, line=collider.lhcb1),
    ele_start='ip1', ele_stop='ip2')

mux = twpresq['mux', 's.ds.l2.b1'] - twpresq['mux', 'e.ds.r1.b1']

assert np.isclose(arc_periodic_solution['lhcb1']['12'].mux[-1],
                  collider.vars['mux12b1']._value, atol=5e-4, rtol=0)
assert np.isclose(arc_periodic_solution['lhcb1']['12'].muy[-1],
                    collider.vars['muy12b1']._value, atol=5e-4, rtol=0)