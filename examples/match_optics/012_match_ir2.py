import xtrack as xt

import xtrack._temp.lhc_match as lm

collider = xt.Multiline.from_json(
    '../../test_data/hllhc15_thick/hllhc15_collider_thick.json')
collider.build_trackers()
collider.vars.load_madx_optics_file(
    "../../test_data/hllhc15_thick/opt_round_150_1500.madx")

default_tol = {None: 1e-8, 'betx': 1e-6, 'bety': 1e-6} # to have no rematching w.r.t. madx

tw_sq_ip1_b1 = lm.propagate_optics_from_beta_star(collider, ip_name='ip1',
        line_name='lhcb1', start='s.ds.r8.b1', end='e.ds.l2.b1',
        beta_star_x=collider.varval['betxip1b1'],
        beta_star_y=collider.varval['betyip1b1'])

mux_compensate = (tw_sq_ip1_b1['mux', 's.ds.l2.b1'] - tw_sq_ip1_b1['mux', 'ip1']
                   - collider.varval['muxip1b1_r'] - collider.varval['mux12b1'])
mux_ir2_target = collider.varval['muxip2b1'] - mux_compensate
muy_compensate = (tw_sq_ip1_b1['muy', 's.ds.l2.b1'] - tw_sq_ip1_b1['muy', 'ip1']
                  - collider.varval['muyip1b1_r'] - collider.varval['muy12b1'])
muy_ir2_target = collider.varval['muyip2b1'] - muy_compensate

arc_periodic_solution = lm.get_arc_periodic_solution(collider)

opt = collider.lhcb1.match(
    solve=False,
    default_tol=default_tol,
    start='s.ds.l2.b1', end='e.ds.r2.b1',
    # Left boundary
    init=tw_sq_ip1_b1,
    targets=[
        # IP optics
        xt.TargetSet(at='ip2',
            betx=collider.varval['betxip2b1'], bety=collider.varval['betyip2b1'],
            alfx=0, alfy=0, dx=0, dpx=0),
        # Right boundary
        xt.TargetList(('betx', 'bety', 'alfx', 'alfy', 'dx', 'dpx'),
                value=arc_periodic_solution['lhcb1']['23'], at='e.ds.r2.b1'),
        xt.TargetRelPhaseAdvance('mux', mux_ir2_target),
        xt.TargetRelPhaseAdvance('muy', muy_ir2_target),
    ],
    vary=xt.VaryList(
        ['kq4.l2b1', 'kq5.l2b1',  'kq6.l2b1', 'kq7.l2b1', 'kq8.l2b1',
         'kq4.r2b1', 'kq5.r2b1',  'kq6.r2b1', 'kq7.r2b1', 'kq8.r2b1',
         'kq9.l2b1', 'kq10.l2b1', 'kqtl11.l2b1', 'kqt12.l2b1', 'kqt13.l2b1',
         'kq9.r2b1', 'kq10.r2b1', 'kqtl11.r2b1', 'kqt12.r2b1', 'kqt13.r2b1']
    )
)

opt.solve()