import pathlib

import numpy as np
import pytest
from cpymad.madx import Madx

import xdeps as xd
import xobjects as xo
import xtrack as xt
import xtrack._temp.lhc_match as lm
from xobjects.test_helpers import for_all_test_contexts, fix_random_seed

test_data_folder = pathlib.Path(
    __file__).parent.joinpath('../test_data').absolute()

@pytest.mark.parametrize(
    'config',
    ['noshift', 'shift']
)
@fix_random_seed(2836475)
def test_lhc_match_phase_15(config):
    test_context = xo.ContextCpu()

    if config == 'noshift':
        d_mux_15_b1 = 0
        d_muy_15_b1 = 0
        d_mux_15_b2 = 0
        d_muy_15_b2 = 0
    elif config == 'shift':
        d_mux_15_b1 = 0.07
        d_muy_15_b1 = 0.05
        d_mux_15_b2 = -0.1
        d_muy_15_b2 = -0.12
    else:
        raise ValueError(f'Invalid config {config}')

    staged_match = True

    collider = xt.load(
        test_data_folder / 'hllhc15_thick/hllhc15_collider_thick.json')
    collider.build_trackers(_context=test_context)
    collider.vars.load_madx_optics_file(
        test_data_folder / "hllhc15_thick/opt_round_150_1500.madx")

    # to have no rematching w.r.t. madx
    default_tol = {None: 1e-8, 'betx': 5e-6, 'bety': 5e-6, 'dx': 1e-7,
                  'alfx': 5e-6, 'alfy': 5e-6}

    lm.set_var_limits_and_steps(collider)

    tw0 = collider.twiss()

    optimizers = {}

    print('Matching ip15 phase:')
    opt = lm.change_phase_non_ats_arcs(collider,
        d_mux_15_b1=d_mux_15_b1, d_muy_15_b1=d_muy_15_b1,
        d_mux_15_b2=d_mux_15_b2, d_muy_15_b2=d_muy_15_b2,
        solve=True, default_tol=default_tol)
    optimizers['phase_15'] = opt

    arc_periodic_solution = lm.get_arc_periodic_solution(collider)

    optimizers.update({'b1':{}, 'b2':{}})

    for bn in ['b1', 'b2']:

        line_name = f'lhc{bn}'

        muxip1_l = collider.varval[f'muxip1{bn}_l']
        muyip1_l = collider.varval[f'muyip1{bn}_l']
        muxip1_r = collider.varval[f'muxip1{bn}_r']
        muyip1_r = collider.varval[f'muyip1{bn}_r']

        muxip5_l = collider.varval[f'muxip5{bn}_l']
        muyip5_l = collider.varval[f'muyip5{bn}_l']
        muxip5_r = collider.varval[f'muxip5{bn}_r']
        muyip5_r = collider.varval[f'muyip5{bn}_r']

        muxip2 = collider.varval[f'muxip2{bn}']
        muyip2 = collider.varval[f'muyip2{bn}']
        muxip4 = collider.varval[f'muxip4{bn}']
        muyip4 = collider.varval[f'muyip4{bn}']
        muxip6 = collider.varval[f'muxip6{bn}']
        muyip6 = collider.varval[f'muyip6{bn}']
        muxip8 = collider.varval[f'muxip8{bn}']
        muyip8 = collider.varval[f'muyip8{bn}']

        mux12 = collider.varval[f'mux12{bn}']
        muy12 = collider.varval[f'muy12{bn}']
        mux45 = collider.varval[f'mux45{bn}']
        muy45 = collider.varval[f'muy45{bn}']
        mux56 = collider.varval[f'mux56{bn}']
        muy56 = collider.varval[f'muy56{bn}']
        mux81 = collider.varval[f'mux81{bn}']
        muy81 = collider.varval[f'muy81{bn}']

        betx_ip1 = collider.varval[f'betxip1{bn}']
        bety_ip1 = collider.varval[f'betyip1{bn}']
        betx_ip5 = collider.varval[f'betxip5{bn}']
        bety_ip5 = collider.varval[f'betyip5{bn}']

        betx_ip2 = collider.varval[f'betxip2{bn}']
        bety_ip2 = collider.varval[f'betyip2{bn}']

        alfx_ip3 = collider.varval[f'alfxip3{bn}']
        alfy_ip3 = collider.varval[f'alfyip3{bn}']
        betx_ip3 = collider.varval[f'betxip3{bn}']
        bety_ip3 = collider.varval[f'betyip3{bn}']
        dx_ip3 = collider.varval[f'dxip3{bn}']
        dpx_ip3 = collider.varval[f'dpxip3{bn}']
        mux_ir3 = collider.varval[f'muxip3{bn}']
        muy_ir3 = collider.varval[f'muyip3{bn}']

        alfx_ip4 = collider.varval[f'alfxip4{bn}']
        alfy_ip4 = collider.varval[f'alfyip4{bn}']
        betx_ip4 = collider.varval[f'betxip4{bn}']
        bety_ip4 = collider.varval[f'betyip4{bn}']
        dx_ip4 = collider.varval[f'dxip4{bn}']
        dpx_ip4 = collider.varval[f'dpxip4{bn}']

        alfx_ip6 = collider.varval[f'alfxip6{bn}']
        alfy_ip6 = collider.varval[f'alfyip6{bn}']
        betx_ip6 = collider.varval[f'betxip6{bn}']
        bety_ip6 = collider.varval[f'betyip6{bn}']
        dx_ip6 = collider.varval[f'dxip6{bn}']
        dpx_ip6 = collider.varval[f'dpxip6{bn}']

        alfx_ip7 = collider.varval[f'alfxip7{bn}']
        alfy_ip7 = collider.varval[f'alfyip7{bn}']
        betx_ip7 = collider.varval[f'betxip7{bn}']
        bety_ip7 = collider.varval[f'betyip7{bn}']
        dx_ip7 = collider.varval[f'dxip7{bn}']
        dpx_ip7 = collider.varval[f'dpxip7{bn}']
        mux_ir7 = collider.varval[f'muxip7{bn}']
        muy_ir7 = collider.varval[f'muyip7{bn}']

        alfx_ip8 = collider.varval[f'alfxip8{bn}']
        alfy_ip8 = collider.varval[f'alfyip8{bn}']
        betx_ip8 = collider.varval[f'betxip8{bn}']
        bety_ip8 = collider.varval[f'betyip8{bn}']
        dx_ip8 = collider.varval[f'dxip8{bn}']
        dpx_ip8 = collider.varval[f'dpxip8{bn}']

        tw_sq_a81_ip1_a12 = lm.propagate_optics_from_beta_star(collider, ip_name='ip1',
                line_name=f'lhc{bn}', start=f's.ds.r8.{bn}', end=f'e.ds.l2.{bn}',
                beta_star_x=betx_ip1, beta_star_y=bety_ip1)

        tw_sq_a45_ip5_a56 = lm.propagate_optics_from_beta_star(collider, ip_name='ip5',
                line_name=f'lhc{bn}', start=f's.ds.r4.{bn}', end=f'e.ds.l6.{bn}',
                beta_star_x=betx_ip5, beta_star_y=bety_ip5)

        (mux_ir2_target, muy_ir2_target, mux_ir4_target, muy_ir4_target,
        mux_ir6_target, muy_ir6_target, mux_ir8_target, muy_ir8_target
            ) = lm.compute_ats_phase_advances_for_auxiliary_irs(line_name,
                tw_sq_a81_ip1_a12, tw_sq_a45_ip5_a56,
                muxip1_l, muyip1_l, muxip1_r, muyip1_r,
                muxip5_l, muyip5_l, muxip5_r, muyip5_r,
                muxip2, muyip2, muxip4, muyip4, muxip6, muyip6, muxip8, muyip8,
                mux12, muy12, mux45, muy45, mux56, muy56, mux81, muy81)

        print(f"Matching IR2 {bn}")
        opt = lm.rematch_ir2(collider, line_name=f'lhc{bn}',
                    boundary_conditions_left=tw_sq_a81_ip1_a12,
                    boundary_conditions_right=arc_periodic_solution[f'lhc{bn}']['23'],
                    mux_ir2=mux_ir2_target, muy_ir2=muy_ir2_target,
                    betx_ip2=betx_ip2, bety_ip2=bety_ip2,
                    solve=True, staged_match=staged_match,
                    default_tol=default_tol)
        optimizers[bn]['ir2'] = opt

        print(f"Matching IR3 {bn}")
        opt = lm.rematch_ir3(collider=collider, line_name=f'lhc{bn}',
                    boundary_conditions_left=arc_periodic_solution[f'lhc{bn}']['23'],
                    boundary_conditions_right=arc_periodic_solution[f'lhc{bn}']['34'],
                    mux_ir3=mux_ir3, muy_ir3=muy_ir3,
                    alfx_ip3=alfx_ip3, alfy_ip3=alfy_ip3,
                    betx_ip3=betx_ip3, bety_ip3=bety_ip3,
                    dx_ip3=dx_ip3, dpx_ip3=dpx_ip3,
                    solve=True, staged_match=staged_match, default_tol=default_tol)
        optimizers[bn]['ir3'] = opt

        print(f"Matching IR4 {bn}")
        opt = lm.rematch_ir4(collider=collider, line_name=f'lhc{bn}',
                    boundary_conditions_left=arc_periodic_solution[f'lhc{bn}']['34'],
                    boundary_conditions_right=tw_sq_a45_ip5_a56,
                    mux_ir4=mux_ir4_target, muy_ir4=muy_ir4_target,
                    alfx_ip4=alfx_ip4, alfy_ip4=alfy_ip4,
                    betx_ip4=betx_ip4, bety_ip4=bety_ip4,
                    dx_ip4=dx_ip4, dpx_ip4=dpx_ip4,
                    solve=True, staged_match=staged_match, default_tol=default_tol)
        optimizers[bn]['ir4'] = opt

        print(f"Matching IP6 {bn}")
        opt = lm.rematch_ir6(collider=collider, line_name=f'lhc{bn}',
                    boundary_conditions_left=tw_sq_a45_ip5_a56,
                    boundary_conditions_right=arc_periodic_solution[f'lhc{bn}']['67'],
                    mux_ir6=mux_ir6_target, muy_ir6=muy_ir6_target,
                    alfx_ip6=alfx_ip6, alfy_ip6=alfy_ip6,
                    betx_ip6=betx_ip6, bety_ip6=bety_ip6,
                    dx_ip6=dx_ip6, dpx_ip6=dpx_ip6,
                    solve=True, staged_match=staged_match, default_tol=default_tol)
        optimizers[bn]['ir6'] = opt

        print(f"Matching IP7 {bn}")
        opt = lm.rematch_ir7(collider=collider, line_name=f'lhc{bn}',
                boundary_conditions_left=arc_periodic_solution[f'lhc{bn}']['67'],
                boundary_conditions_right=arc_periodic_solution[f'lhc{bn}']['78'],
                mux_ir7=mux_ir7, muy_ir7=muy_ir7,
                alfx_ip7=alfx_ip7, alfy_ip7=alfy_ip7,
                betx_ip7=betx_ip7, bety_ip7=bety_ip7,
                dx_ip7=dx_ip7, dpx_ip7=dpx_ip7,
                solve=True, staged_match=staged_match, default_tol=default_tol)
        optimizers[bn]['ir7'] = opt

        print(f"Matching IP8 {bn}")
        opt = lm.rematch_ir8(collider=collider, line_name=f'lhc{bn}',
                boundary_conditions_left=arc_periodic_solution[f'lhc{bn}']['78'],
                boundary_conditions_right=tw_sq_a81_ip1_a12,
                mux_ir8=mux_ir8_target, muy_ir8=muy_ir8_target,
                alfx_ip8=alfx_ip8, alfy_ip8=alfy_ip8,
                betx_ip8=betx_ip8, bety_ip8=bety_ip8,
                dx_ip8=dx_ip8, dpx_ip8=dpx_ip8,
                solve=True, staged_match=staged_match, default_tol=default_tol)
        optimizers[bn]['ir8'] = opt

    if config == 'noshift':
        # Check that the match of the optics did not touch anything
        opts_to_check = []
        opts_to_check.append(optimizers['phase_15']['opt_mq'])
        opts_to_check.append(optimizers['phase_15']['opt_mqt_b1'])
        opts_to_check.append(optimizers['phase_15']['opt_mqt_b2'])
        opts_to_check += optimizers['b1'].values()
        opts_to_check += optimizers['b2'].values()

        for opt in opts_to_check:
            ll = opt.log()
            assert len(ll) >= 2 # will be more than two for staged matches
            assert np.all(ll.vary[-1] == ll.vary[0])

    opt = lm.match_orbit_knobs_ip2_ip8(collider)
    optimizers['orbit_knobs'] = opt

    # Generate madx optics file
    lm.gen_madx_optics_file_auto(collider, f'opt_round_150_1500_xs_{config}.madx')

    tw = collider.twiss()

    # Tunes
    print('Tunes:')
    print(f"  b1: qx={tw.lhcb1.qx:6f} qy={tw.lhcb1.qy:6f}")
    print(f"  b2: qx={tw.lhcb2.qx:6f} qy={tw.lhcb2.qy:6f}")

    print('IP15 phase before:')
    print(f"  b1: d_mux={tw0.lhcb1['mux', 'ip5'] - tw0.lhcb1['mux', 'ip1']:6f} "
        f"      d_muy={tw0.lhcb1['muy', 'ip5'] - tw0.lhcb1['muy', 'ip1']:6f} ")
    print(f"  b2: d_mux={tw0.lhcb2['mux', 'ip5'] - tw0.lhcb2['mux', 'ip1']:6f} "
        f"      d_muy={tw0.lhcb2['muy', 'ip5'] - tw0.lhcb2['muy', 'ip1']:6f} ")

    print('IP15 phase after:')
    print(f"  b1: d_mux={tw.lhcb1['mux', 'ip5'] - tw.lhcb1['mux', 'ip1']:6f} "
        f"      d_muy={tw.lhcb1['muy', 'ip5'] - tw.lhcb1['muy', 'ip1']:6f} ")
    print(f"  b2: d_mux={tw.lhcb2['mux', 'ip5'] - tw.lhcb2['mux', 'ip1']:6f} "
        f"      d_muy={tw.lhcb2['muy', 'ip5'] - tw.lhcb2['muy', 'ip1']:6f} ")

    print('IP15 phase shifts:')
    print(f"  b1: d_mux={tw.lhcb1['mux', 'ip5'] - tw0.lhcb1['mux', 'ip5']:6f} "
                f"d_muy={tw.lhcb1['muy', 'ip5'] - tw0.lhcb1['muy', 'ip5']:6f} ")
    print(f"  b2: d_mux={tw.lhcb2['mux', 'ip5'] - tw0.lhcb2['mux', 'ip5']:6f} "
                f"d_muy={tw.lhcb2['muy', 'ip5'] - tw0.lhcb2['muy', 'ip5']:6f} ")

    collider_ref = xt.load(
        test_data_folder / 'hllhc15_thick/hllhc15_collider_thick.json')
    collider_ref.build_trackers()
    collider_ref.vars.load_madx_optics_file(
        test_data_folder / "hllhc15_thick/opt_round_150_1500.madx")

    tw = collider.twiss()
    tw_ref = collider_ref.twiss()

    d_mux = {'lhcb1': d_mux_15_b1, 'lhcb2': d_mux_15_b2}
    d_muy = {'lhcb1': d_muy_15_b1, 'lhcb2': d_muy_15_b2}

    for ll in ['lhcb1', 'lhcb2']:

        xo.assert_allclose(tw[ll].qx, tw_ref[ll].qx, atol=1e-7, rtol=0)
        xo.assert_allclose(tw[ll].qy, tw_ref[ll].qy, atol=1e-7, rtol=0)

        mux_15 = tw[ll]['mux', 'ip5'] - tw[ll]['mux', 'ip1']
        mux_ref_15 = tw_ref[ll]['mux', 'ip5'] - tw_ref[ll]['mux', 'ip1']

        muy_15 = tw[ll]['muy', 'ip5'] - tw[ll]['muy', 'ip1']
        muy_ref_15 = tw_ref[ll]['muy', 'ip5'] - tw_ref[ll]['muy', 'ip1']

        xo.assert_allclose(mux_15, mux_ref_15 + d_mux[ll], atol=1e-7, rtol=0)
        xo.assert_allclose(muy_15, muy_ref_15 + d_muy[ll], atol=1e-7, rtol=0)

        twip = tw[ll].rows['ip.*']
        twip_ref = tw_ref[ll].rows['ip.*']

        xo.assert_allclose(twip['betx'], twip_ref['betx'], rtol=1e-6, atol=0)
        xo.assert_allclose(twip['bety'], twip_ref['bety'], rtol=1e-6, atol=0)
        xo.assert_allclose(twip['alfx'], twip_ref['alfx'], rtol=0, atol=1e-5)
        xo.assert_allclose(twip['alfy'], twip_ref['alfy'], rtol=0, atol=1e-5)
        xo.assert_allclose(twip['dx'], twip_ref['dx'], rtol=0, atol=1e-6)
        xo.assert_allclose(twip['dy'], twip_ref['dy'], rtol=0, atol=1e-6)
        xo.assert_allclose(twip['dpx'], twip_ref['dpx'], rtol=0, atol=1e-7)
        xo.assert_allclose(twip['dpy'], twip_ref['dpy'], rtol=0, atol=1e-7)


    # -----------------------------------------------------------------------------

    # Check higher level knobs

    collider.vars['on_x2'] = 34
    tw = collider.twiss()
    collider.vars['on_x2'] = 0

    xo.assert_allclose(tw.lhcb1['py', 'ip2'], 34e-6, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb2['py', 'ip2'], -34e-6, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb1['y', 'ip2'], 0, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb2['y', 'ip2'], 0, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb1['px', 'ip2'], 0, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb2['px', 'ip2'], 0, atol=1e-8, rtol=0)

    collider.vars['on_x8'] = 35
    tw = collider.twiss()
    collider.vars['on_x8'] = 0

    xo.assert_allclose(tw.lhcb1['px', 'ip8'], 35e-6, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb2['px', 'ip8'], -35e-6, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb1['x', 'ip8'], 0, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb2['x', 'ip8'], 0, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb1['py', 'ip8'], 0, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb2['py', 'ip8'], 0, atol=1e-8, rtol=0)

    collider.vars['on_sep2'] = 0.5
    tw = collider.twiss()
    collider.vars['on_sep2'] = 0

    xo.assert_allclose(tw.lhcb1['x', 'ip2'], -0.5e-3, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb2['x', 'ip2'], 0.5e-3, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb1['px', 'ip2'], 0, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb2['px', 'ip2'], 0, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb1['y', 'ip2'], 0, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb2['y', 'ip2'], 0, atol=1e-8, rtol=0)

    collider.vars['on_sep8'] = 0.6
    tw = collider.twiss()
    collider.vars['on_sep8'] = 0

    xo.assert_allclose(tw.lhcb1['y', 'ip8'], 0.6e-3, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb2['y', 'ip8'], -0.6e-3, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb1['py', 'ip8'], 0, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb2['py', 'ip8'], 0, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb1['x', 'ip8'], 0, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb2['x', 'ip8'], 0, atol=1e-8, rtol=0)

    # Check lower level knobs (disconnects higher level knobs)

    collider.vars['on_o2v'] = 0.3
    tw = collider.twiss()
    collider.vars['on_o2v'] = 0

    xo.assert_allclose(tw.lhcb1['y', 'ip2'], 0.3e-3, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb2['y', 'ip2'], 0.3e-3, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb1['py', 'ip2'], 0., atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb2['py', 'ip2'], 0., atol=1e-8, rtol=0)

    collider.vars['on_o2h'] = 0.4
    tw = collider.twiss()
    collider.vars['on_o2h'] = 0

    xo.assert_allclose(tw.lhcb1['x', 'ip2'], 0.4e-3, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb2['x', 'ip2'], 0.4e-3, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb1['px', 'ip2'], 0., atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb2['px', 'ip2'], 0., atol=1e-8, rtol=0)

    collider.vars['on_o8v'] = 0.5
    tw = collider.twiss()
    collider.vars['on_o8v'] = 0

    xo.assert_allclose(tw.lhcb1['y', 'ip8'], 0.5e-3, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb2['y', 'ip8'], 0.5e-3, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb1['py', 'ip8'], 0., atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb2['py', 'ip8'], 0., atol=1e-8, rtol=0)

    collider.vars['on_o8h'] = 0.6
    tw = collider.twiss()
    collider.vars['on_o8h'] = 0

    xo.assert_allclose(tw.lhcb1['x', 'ip8'], 0.6e-3, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb2['x', 'ip8'], 0.6e-3, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb1['px', 'ip8'], 0., atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb2['px', 'ip8'], 0., atol=1e-8, rtol=0)

    collider.vars['on_a2h'] = 20
    tw = collider.twiss()
    collider.vars['on_a2h'] = 0

    xo.assert_allclose(tw.lhcb1['x', 'ip2'], 0, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb2['x', 'ip2'], 0, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb1['px', 'ip2'], 20e-6, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb2['px', 'ip2'], 20e-6, atol=1e-8, rtol=0)

    collider.vars['on_a2v'] = 15
    tw = collider.twiss()
    collider.vars['on_a2v'] = 0

    xo.assert_allclose(tw.lhcb1['y', 'ip2'], 0, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb2['y', 'ip2'], 0, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb1['py', 'ip2'], 15e-6, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb2['py', 'ip2'], 15e-6, atol=1e-8, rtol=0)

    collider.vars['on_a8h'] = 20
    tw = collider.twiss()
    collider.vars['on_a8h'] = 0

    xo.assert_allclose(tw.lhcb1['x', 'ip8'], 0, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb2['x', 'ip8'], 0, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb1['px', 'ip8'], 20e-6, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb2['px', 'ip8'], 20e-6, atol=1e-8, rtol=0)

    collider.vars['on_a8v'] = 50
    tw = collider.twiss()
    collider.vars['on_a8v'] = 0

    xo.assert_allclose(tw.lhcb1['y', 'ip8'], 0, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb2['y', 'ip8'], 0, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb1['py', 'ip8'], 50e-6, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb2['py', 'ip8'], 50e-6, atol=1e-8, rtol=0)

    collider.vars['on_x2v'] = 100
    tw = collider.twiss()
    collider.vars['on_x2v'] = 0

    xo.assert_allclose(tw.lhcb1['y', 'ip2'], 0, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb2['y', 'ip2'], 0, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb1['py', 'ip2'], 100e-6, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb2['py', 'ip2'], -100e-6, atol=1e-8, rtol=0)

    collider.vars['on_x2h'] = 120
    tw = collider.twiss()
    collider.vars['on_x2h'] = 0

    xo.assert_allclose(tw.lhcb1['x', 'ip2'], 0, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb2['x', 'ip2'], 0, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb1['px', 'ip2'], 120e-6, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb2['px', 'ip2'], -120e-6, atol=1e-8, rtol=0)


    collider.vars['on_x8h'] = 100
    tw = collider.twiss()
    collider.vars['on_x8h'] = 0

    xo.assert_allclose(tw.lhcb1['x', 'ip8'], 0, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb2['x', 'ip8'], 0, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb1['px', 'ip8'], 100e-6, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb2['px', 'ip8'], -100e-6, atol=1e-8, rtol=0)

    collider.vars['on_x8v'] = 120
    tw = collider.twiss()
    collider.vars['on_x8v'] = 0

    xo.assert_allclose(tw.lhcb1['y', 'ip8'], 0, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb2['y', 'ip8'], 0, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb1['py', 'ip8'], 120e-6, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb2['py', 'ip8'], -120e-6, atol=1e-8, rtol=0)

    collider.vars['on_sep2h'] = 1.6
    tw = collider.twiss()
    collider.vars['on_sep2h'] = 0

    xo.assert_allclose(tw.lhcb1['x', 'ip2'], 1.6e-3, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb2['x', 'ip2'], -1.6e-3, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb1['px', 'ip2'], 0, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb2['px', 'ip2'], 0, atol=1e-8, rtol=0)

    collider.vars['on_sep2v'] = 1.7
    tw = collider.twiss()
    collider.vars['on_sep2v'] = 0

    xo.assert_allclose(tw.lhcb1['y', 'ip2'], 1.7e-3, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb2['y', 'ip2'], -1.7e-3, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb1['py', 'ip2'], 0, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb2['py', 'ip2'], 0, atol=1e-8, rtol=0)

    collider.vars['on_sep8h'] = 1.5
    tw = collider.twiss()
    collider.vars['on_sep8h'] = 0

    xo.assert_allclose(tw.lhcb1['x', 'ip8'], 1.5e-3, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb2['x', 'ip8'], -1.5e-3, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb1['px', 'ip8'], 0, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb2['px', 'ip8'], 0, atol=1e-8, rtol=0)

    collider.vars['on_sep8v'] = 1.7
    tw = collider.twiss()
    collider.vars['on_sep8v'] = 0

    xo.assert_allclose(tw.lhcb1['y', 'ip8'], 1.7e-3, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb2['y', 'ip8'], -1.7e-3, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb1['py', 'ip8'], 0, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb2['py', 'ip8'], 0, atol=1e-8, rtol=0)

    # Both knobs together
    collider.vars['on_x8h'] = 120
    collider.vars['on_sep8h'] = 1.7
    tw = collider.twiss()
    collider.vars['on_x8h'] = 0
    collider.vars['on_sep8h'] = 0

    xo.assert_allclose(tw.lhcb1['x', 'ip8'], 1.7e-3, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb2['x', 'ip8'], -1.7e-3, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb1['px', 'ip8'], 120e-6, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.lhcb2['px', 'ip8'], -120e-6, atol=1e-8, rtol=0)

    # Check generated optics in madx

    mad=Madx(stdout=False)
    mad.call(str(test_data_folder / 'hllhc15_thick/lhc.seq'))
    mad.call(str(test_data_folder / 'hllhc15_thick/hllhc_sequence.madx'))
    mad.input('beam, sequence=lhcb1, particle=proton, energy=7000;')
    mad.use('lhcb1')
    mad.input('beam, sequence=lhcb2, particle=proton, energy=7000, bv=-1;')
    mad.use('lhcb2')
    mad.call(f"opt_round_150_1500_xs_{config}.madx")

    mad.input('twiss, sequence=lhcb1, table=twb1')
    mad.input('twiss, sequence=lhcb2, table=twb2')
    twmad_b1 = xd.Table(mad.table.twb1)
    twmad_b2 = xd.Table(mad.table.twb2)

    xo.assert_allclose(twmad_b1['betx', 'ip1:1'], 0.15, rtol=1e-7, atol=0)
    xo.assert_allclose(twmad_b1['bety', 'ip1:1'], 0.15, rtol=1e-7, atol=0)
    xo.assert_allclose(twmad_b2['betx', 'ip1:1'], 0.15, rtol=1e-7, atol=0)
    xo.assert_allclose(twmad_b2['bety', 'ip1:1'], 0.15, rtol=1e-7, atol=0)

    twmad_b1.rows['ip.*'].cols['betx bety x y px py'].show()
    twmad_b2.rows['ip.*'].cols['betx bety x y px py'].show()

    # Test orbit knobs
    mad.globals.on_x8 = 100
    mad.globals.on_x2 = 110

    mad.input('twiss, sequence=lhcb1, table=twb1')
    mad.input('twiss, sequence=lhcb2, table=twb2')
    twmad_b1 = xd.Table(mad.table.twb1)
    twmad_b2 = xd.Table(mad.table.twb2)

    xo.assert_allclose(twmad_b1['px', 'ip8:1'], 100e-6, rtol=0, atol=1e-9)
    xo.assert_allclose(twmad_b2['px', 'ip8:1'], -100e-6, rtol=0, atol=1e-9)
    xo.assert_allclose(twmad_b1['py', 'ip2:1'], 110e-6, rtol=0, atol=5e-10)
    xo.assert_allclose(twmad_b2['py', 'ip2:1'], -110e-6, rtol=0, atol=5e-10)

    # Match tunes and chromaticity in the Xsuite model
    opt = collider.match(
        solve=False,
        vary=[
            xt.VaryList(['kqtf.b1', 'kqtd.b1', 'ksf.b1', 'ksd.b1'], step=1e-7),
            xt.VaryList(['kqtf.b2', 'kqtd.b2', 'ksf.b2', 'ksd.b2'], step=1e-7)],
        targets = [
            xt.TargetSet(line='lhcb1', qx=62.315, qy=60.325, tol=1e-6),
            xt.TargetSet(line='lhcb1', dqx=10.0, dqy=12.0, tol=1e-4),
            xt.TargetSet(line='lhcb2', qx=62.316, qy=60.324, tol=1e-6),
            xt.TargetSet(line='lhcb2', dqx=9.0, dqy=11.0, tol=1e-4)])
    opt.solve()

    # Transfer knobs to madx model and check matched values

    for kk, vv in opt.get_knob_values().items():
        mad.globals[kk] = vv

    mad.input('twiss, sequence=lhcb1, table=twb1')
    mad.input('twiss, sequence=lhcb2, table=twb2')

    xo.assert_allclose(mad.table.twb1.summary.q1, 62.315, rtol=0, atol=1e-6)
    xo.assert_allclose(mad.table.twb1.summary.q2, 60.325, rtol=0, atol=1e-6)
    xo.assert_allclose(mad.table.twb2.summary.q1, 62.316, rtol=0, atol=1e-6)
    xo.assert_allclose(mad.table.twb2.summary.q2, 60.324, rtol=0, atol=1e-6)
    xo.assert_allclose(mad.table.twb1.summary.dq1, 10.0, rtol=0, atol=0.5)
    xo.assert_allclose(mad.table.twb1.summary.dq2, 12.0, rtol=0, atol=0.5)
    xo.assert_allclose(mad.table.twb2.summary.dq1, 9.0, rtol=0, atol=0.5)
    xo.assert_allclose(mad.table.twb2.summary.dq2, 11.0, rtol=0, atol=0.5)
