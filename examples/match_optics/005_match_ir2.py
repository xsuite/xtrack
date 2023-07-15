import xtrack as xt
import lhc_match as lm

collider = xt.Multiline.from_json('hllhc.json')
collider.build_trackers()
collider.vars.load_madx_optics_file(
    "../../test_data/hllhc15_thick/opt_round_150_1500.madx")

default_tol = {None: 1e-8, 'betx': 1e-6, 'bety': 1e-6} # to have no rematching w.r.t. madx

arc_periodic_solution = lm.get_arc_periodic_solution(collider)
tw_sq_ip1_b1 = lm.propagate_optics_from_beta_star(collider, ip_name='ip1',
        line_name='lhcb1', ele_start='s.ds.r8.b1', ele_stop='e.ds.l2.b1',
        beta_star_x=0.15, beta_star_y = 0.15)

mux_compensate = (tw_sq_ip1_b1['mux', 's.ds.l2.b1'] - tw_sq_ip1_b1['mux', 'ip1']
                   - collider.varval['muxip1b1_r'] - collider.varval['mux12b1'])
mux_ir2_target = collider.varval['muxip2b1'] - mux_compensate

opt = collider.lhcb1.match(
    solve=False,
    default_tol=default_tol,
    ele_start='s.ds.l2.b1', ele_stop='e.ds.r2.b1',
    # Left boundary
    twiss_init='preserve_start', table_for_twiss_init=tw_sq_ip1_b1,
    targets=[
        # IP optics
        xt.TargetList(('alfx', 'alfy', 'dx', 'dpx'), value=0, at='ip2'),
        xt.Target('betx', value=collider.varval['betxip2b1'], at='ip2'),
        xt.Target('bety', value=collider.varval['betyip2b1'], at='ip2'),
        # Right boundary
        xt.TargetList(('betx', 'bety', 'alfx', 'alfy', 'dx', 'dpx'),
                value=arc_periodic_solution['lhcb1']['23'], at='e.ds.r2.b1'),
        xt.TargetPhaseAdvance('mux', mux_ir2_target),
    ],
    vary=xt.VaryList(
        ['kq4.l2b1', 'kq5.l2b1',  'kq6.l2b1', 'kq7.l2b1', 'kq8.l2b1',
         'kq4.r2b1', 'kq5.r2b1',  'kq6.r2b1', 'kq7.r2b1', 'kq8.r2b1',
         'kq9.l2b1', 'kq10.l2b1', 'kqtl11.l2b1', 'kqt12.l2b1', 'kqt13.l2b1',
         'kq9.r2b1', 'kq10.r2b1', 'kqtl11.r2b1', 'kqt12.r2b1', 'kqt13.r2b1']
    )
)

opt.solve()