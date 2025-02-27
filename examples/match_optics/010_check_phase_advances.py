import numpy as np
import xtrack as xt

import xtrack._temp.lhc_match as lm

collider = xt.Environment.from_json('hllhc.json')
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


tw_presq_ip1_b1 = lm.propagate_optics_from_beta_star(collider, ip_name='ip1',
        line_name='lhcb1', start='s.ds.r8.b1', end='e.ds.l2.b1',
        beta_star_x=0.5, beta_star_y = 0.5)

assert np.isclose(
    tw_presq_ip1_b1['mux', 'ip1'] -  tw_presq_ip1_b1['mux', 's.ds.l1.b1'],
    collider.vars['muxip1b1_l']._value, atol=1e-10, rtol=0)
assert np.isclose(
    tw_presq_ip1_b1['muy', 'ip1'] -  tw_presq_ip1_b1['muy', 's.ds.l1.b1'],
    collider.vars['muyip1b1_l']._value, atol=1e-10, rtol=0)
assert np.isclose(
    tw_presq_ip1_b1['mux', 'e.ds.r1.b1'] -  tw_presq_ip1_b1['mux', 'ip1'],
    collider.vars['muxip1b1_r']._value, atol=1e-10, rtol=0)
assert np.isclose(
    tw_presq_ip1_b1['muy', 'e.ds.r1.b1'] -  tw_presq_ip1_b1['muy', 'ip1'],
    collider.vars['muyip1b1_r']._value, atol=1e-10, rtol=0)

# Beam 1

tw_sq_ip1_b1 = lm.propagate_optics_from_beta_star(collider, ip_name='ip1',
        line_name='lhcb1', start='s.ds.r7.b1', end='e.ds.l3.b1',
        beta_star_x=0.15, beta_star_y = 0.15)
vt = collider.vars.get_table()

# Check that the phase advance the whole ATS section is preserved
assert np.isclose(
        tw_sq_ip1_b1['mux', 'e.ds.r2.b1'] - tw_sq_ip1_b1['mux', 's.ds.l8.b1'],
        (vt['value', 'muxip8b1']+vt['value', 'mux81b1']+vt['value', 'muxip1b1']
            + vt['value', 'mux12b1'] +  vt['value', 'muxip2b1']),
        atol=1e-9, rtol=0)

# Change of arc phase advance wrt periodic solution
dmux_81 = ((tw_sq_ip1_b1['mux', 's.ds.l1.b1'] - tw_sq_ip1_b1['mux', 'e.ds.r8.b1'])
           - vt['value', 'mux81b1'])

# Change ip1_l phase advance wrt presquezed solution
dmux_ip1_l = (tw_sq_ip1_b1['mux', 'ip1'] - tw_sq_ip1_b1['mux', 's.ds.l1.b1']
              - vt['value', 'muxip1b1_l'])

muxip8b1_sq = (tw_sq_ip1_b1['mux', 'e.ds.r8.b1'] - tw_sq_ip1_b1['mux', 's.ds.l8.b1'])

assert np.isclose(muxip8b1_sq - vt['value', 'muxip8b1'], -(dmux_81 + dmux_ip1_l),
                  atol=1e-9, rtol=0)


# Beam 2
tw_sq_ip1_b2 = lm.propagate_optics_from_beta_star(collider, ip_name='ip1',
        line_name='lhcb2', start='s.ds.r7.b2', end='e.ds.l3.b2',
        beta_star_x=0.15, beta_star_y = 0.15)







