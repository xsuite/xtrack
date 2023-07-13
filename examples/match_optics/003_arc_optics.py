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


# ip_name = 'ip5'
# ele_start = 's.ds.r4.b1'
# ele_stop = 'e.ds.l6.b1'

ip_name = 'ip1'
line_name = 'lhcb1'
ele_start = 's.ds.r8.b1'
ele_stop = 'e.ds.l2.b1'

ip_name = 'ip1'
line_name = 'lhcb2'
ele_start = 's.ds.r8.b2'
ele_stop = 'e.ds.l2.b2'

beta_star_x = 0.5
beta_star_y = 0.5


assert collider.lhcb1.twiss_default.get('reverse', False) is False
assert collider.lhcb2.twiss_default['reverse'] is True
assert collider.lhcb1.element_names[1] == 'ip1'
assert collider.lhcb2.element_names[1] == 'ip1.l1'
assert collider.lhcb1.element_names[-2] == 'ip1.l1'
assert collider.lhcb2.element_names[-2] == 'ip1'


if ip_name == 'ip1':
    ele_stop_left = 'ip1.l1'
    ele_start_right = 'ip1'
else:
    ele_stop_left = ip_name
    ele_start_right = ip_name

tw_left = collider[line_name].twiss(ele_start=ele_start, ele_stop=ele_stop_left,
                    twiss_init=xt.TwissInit(line=collider[line_name],
                                            element_name=ele_stop_left,
                                            betx=beta_star_x,
                                            bety=beta_star_y))
tw_right = collider[line_name].twiss(ele_start=ele_start_right, ele_stop=ele_stop,
                    twiss_init=xt.TwissInit(line=collider[line_name],
                                            element_name=ele_start_right,
                                            betx=beta_star_x,
                                            bety=beta_star_y))

tw_ip = xt.TwissTable.concatenate([tw_left, tw_right])



