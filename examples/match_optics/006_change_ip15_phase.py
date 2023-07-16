import xtrack as xt
import lhc_match as lm

default_tol = {None: 1e-8, 'betx': 1e-6, 'bety': 1e-6} # to have no rematching w.r.t. madx

collider = xt.Multiline.from_json('hllhc.json')
collider.build_trackers()
collider.vars.load_madx_optics_file(
    "../../test_data/hllhc15_thick/opt_round_150_1500.madx")

d_mux_15_b1 = None
d_muy_15_b1 = None
d_mux_15_b2 = None
d_muy_15_b2 = None

d_mux_15_b1 = 0#.1
d_muy_15_b1 = 0#.12
# d_mux_15_b2 = -0.09
# d_muy_15_b2 = -0.15

staged_match = False

opt = lm.change_phase_non_ats_arcs(collider,
    d_mux_15_b1=d_mux_15_b1, d_muy_15_b1=d_muy_15_b1,
    d_mux_15_b2=d_mux_15_b2, d_muy_15_b2=d_muy_15_b2,
    solve=True, default_tol=default_tol)

arc_periodic_solution = lm.get_arc_periodic_solution(collider)

for bn in ['b1', 'b2']:

    print(f"Matching IR2 {bn}")

    tw_sq_ip1 = lm.propagate_optics_from_beta_star(collider, ip_name='ip1',
            line_name=f'lhc{bn}', ele_start=f's.ds.r8.{bn}', ele_stop=f'e.ds.l2.{bn}',
            beta_star_x=collider.varval[f'betxip1{bn}'],
            beta_star_y=collider.varval[f'betyip1{bn}'])

    mux_compensate = (tw_sq_ip1['mux', f's.ds.l2.{bn}'] - tw_sq_ip1['mux', 'ip1']
                    - collider.varval[f'muxip1{bn}_r'] - collider.varval[f'mux12{bn}'])
    mux_ir2_target = collider.varval[f'muxip2{bn}'] - mux_compensate

    muy_compensate = (tw_sq_ip1['muy', f's.ds.l2.{bn}'] - tw_sq_ip1['muy', 'ip1']
                    - collider.varval[f'muyip1{bn}_r'] - collider.varval[f'muy12{bn}'])
    muy_ir2_target = collider.varval[f'muyip2{bn}'] - muy_compensate


    betx_ip2 = collider.varval[f'betxip2{bn}']
    bety_ip2 = collider.varval[f'betyip2{bn}']

    opt = lm.rematch_ir2(collider, line_name=f'lhc{bn}',
                boundary_conditions_left=tw_sq_ip1,
                boundary_conditions_right=arc_periodic_solution[f'lhc{bn}']['23'],
                mux_ir2=mux_ir2_target, muy_ir2=muy_ir2_target,
                betx_ip2=betx_ip2, bety_ip2=bety_ip2,
                solve=True, staged_match=staged_match,
                default_tol=default_tol)

    print(f"Matching IR2 {bn}")

    boundary_conditions_left = arc_periodic_solution[f'lhc{bn}']['23']
    boundary_conditions_right = arc_periodic_solution[f'lhc{bn}']['34']

    alfx_ip3 = collider.varval[f'alfxip3{bn}']
    alfy_ip3 = collider.varval[f'alfyip3{bn}']
    betx_ip3 = collider.varval[f'betxip3{bn}']
    bety_ip3 = collider.varval[f'betyip3{bn}']
    dx_ip3 = collider.varval[f'dxip3{bn}']
    dpx_ip3 = collider.varval[f'dpxip3{bn}']
    mux_ir3 = collider.varval[f'muxip3{bn}']
    muy_ir3 = collider.varval[f'muyip3{bn}']

    opt = collider[f'lhc{bn}'].match(
        solve=False,
        default_tol=default_tol,
        ele_start=f's.ds.l3.{bn}', ele_stop=f'e.ds.r3.{bn}',
        # Left boundary
        twiss_init='preserve_start', table_for_twiss_init=boundary_conditions_left,
        targets=[
            xt.Target('alfx', alfx_ip3, at='ip3'),
            xt.Target('alfy', alfy_ip3, at='ip3'),
            xt.Target('betx', betx_ip3, at='ip3'),
            xt.Target('bety', bety_ip3, at='ip3'),
            xt.Target('dx', dx_ip3, at='ip3'),
            xt.Target('dpx', dpx_ip3, at='ip3'),
            xt.TargetList(('betx', 'bety', 'alfx', 'alfy', 'dx', 'dpx'),
                    value=boundary_conditions_right, at=f'e.ds.r3.{bn}'),
            xt.TargetRelPhaseAdvance('mux', mux_ir3),
            xt.TargetRelPhaseAdvance('muy', muy_ir3),
        ],
        vary=[
            xt.VaryList([f'kqt13.l3{bn}', f'kqt12.l3{bn}', f'kqtl11.l3{bn}',
                         f'kqtl10.l3{bn}', f'kqtl9.l3{bn}', f'kqtl8.l3{bn}',
                         f'kqtl7.l3{bn}', f'kq6.l3{bn}',
                         f'kq6.r3{bn}', f'kqtl7.r3{bn}',
                         f'kqtl8.r3{bn}', f'kqtl9.r3{bn}', f'kqtl10.r3{bn}',
                         f'kqtl11.r3{bn}', f'kqt12.r3{bn}', f'kqt13.r3{bn}'])
        ]
    )

    opt.solve()
