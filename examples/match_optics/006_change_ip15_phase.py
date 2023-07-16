import xtrack as xt
import lhc_match as lm

collider = xt.Multiline.from_json('hllhc.json')
collider.build_trackers()
collider.vars.load_madx_optics_file(
    "../../test_data/hllhc15_thick/opt_round_150_1500.madx")

d_mux_15_b1 = None
d_muy_15_b1 = None
d_mux_15_b2 = None
d_muy_15_b2 = None

d_mux_15_b1 = 0.1
d_muy_15_b1 = 0.12
# d_mux_15_b2 = -0.09
# d_muy_15_b2 = -0.15

staged_match = False

opt = lm.change_phase_non_ats_arcs(collider,
    d_mux_15_b1=d_mux_15_b1, d_muy_15_b1=d_muy_15_b1,
    d_mux_15_b2=d_mux_15_b2, d_muy_15_b2=d_muy_15_b2,
    solve=True)

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

    arc_periodic_solution = lm.get_arc_periodic_solution(collider, arc_name=['23'])
    bet_ip2_x = collider.varval[f'betxip2{bn}']
    bet_ip2_y = collider.varval[f'betyip2{bn}']
    boundary_conditions_left = tw_sq_ip1
    boundary_conditions_right = arc_periodic_solution[f'lhc{bn}']['23']

    opt = collider[f'lhc{bn}'].match(
        solve=False,
        ele_start=f's.ds.l2.{bn}', ele_stop=f'e.ds.r2.{bn}',
        # Left boundary
        twiss_init='preserve_start', table_for_twiss_init=boundary_conditions_left,
        targets=[
            # IP optics
            xt.TargetList(('alfx', 'alfy', 'dx', 'dpx'), value=0, at='ip2',    tag='stage2'),
            xt.Target('betx', value=bet_ip2_x, at='ip2', tag='stage2'),
            xt.Target('bety', value=bet_ip2_y, at='ip2', tag='stage2'),
            # Right boundary
            xt.TargetList(('betx', 'bety', 'alfx', 'alfy', 'dx', 'dpx'), tag='stage0',
                    value=boundary_conditions_right, at=f'e.ds.r2.{bn}'),
            xt.TargetRelPhaseAdvance('mux', mux_ir2_target, tag='stage0'),
            xt.TargetRelPhaseAdvance('muy', muy_ir2_target, tag='stage0'),
        ],
        vary=[
            xt.VaryList([
                f'kq9.l2{bn}', f'kq10.l2{bn}', f'kqtl11.l2{bn}', f'kqt12.l2{bn}', f'kqt13.l2{bn}',
                f'kq9.r2{bn}', f'kq10.r2{bn}', f'kqtl11.r2{bn}', f'kqt12.r2{bn}', f'kqt13.r2{bn}'],
                tag='stage0'),
            xt.VaryList(
                [f'kq4.l2{bn}', f'kq5.l2{bn}',  f'kq6.l2{bn}',    f'kq7.l2{bn}',   f'kq8.l2{bn}',
                 f'kq6.r2{bn}',    f'kq7.r2{bn}',   f'kq8.r2{bn}'],
                tag='stage1'),
            xt.VaryList([f'kq4.r2{bn}', f'kq5.r2{bn}'], tag='stage2')
        ]
    )

    if staged_match:
        opt.disable_all_vary()
        opt.disable_all_targets()

        opt.enable_vary(tag='stage0')
        opt.enable_targets(tag='stage0')
        opt.solve()

        opt.enable_vary(tag='stage1')
        opt.solve()

        opt.enable_vary(tag='stage2')
        opt.enable_targets(tag='stage2')
        opt.solve()
    else:
        opt.solve()

# Check
tw_sq_check = lm.propagate_optics_from_beta_star(collider, ip_name='ip1',
        line_name='lhcb1', ele_start='e.ds.r8.b1', ele_stop='s.ds.r3.b1',
        beta_star_x=collider.varval['betxip1b1'],
        beta_star_y=collider.varval['betyip1b1'])