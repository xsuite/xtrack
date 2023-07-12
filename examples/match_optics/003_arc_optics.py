import numpy as np
import xtrack as xt

import lhc_match as lm

collider = xt.Multiline.from_json('hllhc.json')
collider.build_trackers()
collider.vars.load_madx_optics_file(
    "../../test_data/hllhc15_thick/opt_round_150_1500.madx")

arc_periodic_solution =lm.get_arc_periodic_solution(collider)

for aa in ['12', '23', '34', '45', '56', '67', '78', '81']:
    assert np.isclose(arc_periodic_solution['lhcb1'][aa].mux[-1],
                        collider.vars[f'mux{aa}b1']._value, atol=1e-10, rtol=0)
    assert np.isclose(arc_periodic_solution['lhcb1'][aa].muy[-1],
                        collider.vars[f'muy{aa}b1']._value, atol=1e-10, rtol=0)
    assert np.isclose(arc_periodic_solution['lhcb2'][aa].mux[-1],
                        collider.vars[f'mux{aa}b2']._value, atol=1e-10, rtol=0)
    assert np.isclose(arc_periodic_solution['lhcb2'][aa].muy[-1],
                        collider.vars[f'muy{aa}b2']._value, atol=1e-10, rtol=0)
