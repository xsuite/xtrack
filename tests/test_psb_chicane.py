import json
import pathlib

import numpy as np
import pandas as pd
from cpymad.madx import Madx

import xdeps as xd
import xobjects as xo
import xpart as xp
import xtrack as xt
from xobjects.test_helpers import for_all_test_contexts
from xtrack.slicing import Teapot, Strategy

test_data_folder = pathlib.Path(
        __file__).parent.joinpath('../test_data').absolute()

@for_all_test_contexts
def test_psb_chicane(test_context):
    mad = Madx(stdout=False)

    # Load mad model and apply element shifts
    mad.input(f'''
    call, file = '{str(test_data_folder)}/psb_chicane/psb.seq';
    call, file = '{str(test_data_folder)}/psb_chicane/psb_fb_lhc.str';

    beam, particle=PROTON, pc=0.5708301551893517;
    use, sequence=psb1;

    select,flag=error,clear;
    select,flag=error,pattern=bi1.bsw1l1.1*;
    ealign, dx=-0.0057;

    select,flag=error,clear;
    select,flag=error,pattern=bi1.bsw1l1.2*;
    select,flag=error,pattern=bi1.bsw1l1.3*;
    select,flag=error,pattern=bi1.bsw1l1.4*;
    ealign, dx=-0.0442;

    twiss;
    ''')

    line = xt.Line.from_madx_sequence(mad.sequence.psb1,
                                allow_thick=True,
                                enable_align_errors=True,
                                deferred_expressions=True)
    line.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV,
                                gamma0=mad.sequence.psb1.beam.gamma)
    line.twiss_default['method'] = '4d'
    line.configure_bend_model(core='adaptive')

    # Build chicane knob (k0)
    line.vars['bsw_k0l'] = 0
    line.vars['k0bi1bsw1l11'] = (line.vars['bsw_k0l'] / line['bi1.bsw1l1.1'].length)
    line.vars['k0bi1bsw1l12'] = (-line.vars['bsw_k0l'] / line['bi1.bsw1l1.2'].length)
    line.vars['k0bi1bsw1l13'] = (-line.vars['bsw_k0l'] / line['bi1.bsw1l1.3'].length)
    line.vars['k0bi1bsw1l14'] = (line.vars['bsw_k0l'] / line['bi1.bsw1l1.4'].length)


    # Build knob to model eddy currents (k2)
    line.vars['bsw_k2l'] = 0
    line.element_refs['bi1.bsw1l1.1'].knl[2] = line.vars['bsw_k2l']
    line.element_refs['bi1.bsw1l1.2'].knl[2] = -line.vars['bsw_k2l']
    line.element_refs['bi1.bsw1l1.3'].knl[2] = -line.vars['bsw_k2l']
    line.element_refs['bi1.bsw1l1.4'].knl[2] = line.vars['bsw_k2l']

    # Test to_dict roundtrip
    line = xt.Line.from_dict(line.to_dict())
    line.build_tracker(_context=test_context)

    bsw_k2l_ref = -9.7429e-2
    bsw_k0l_ref = 6.6e-2

    line.vars['bsw_k2l'] = bsw_k2l_ref
    line.vars['bsw_k0l'] = bsw_k0l_ref
    xo.assert_allclose(line['bi1.bsw1l1.1']._xobject.knl[2], bsw_k2l_ref, rtol=0, atol=1e-10)
    xo.assert_allclose(line['bi1.bsw1l1.2']._xobject.knl[2], -bsw_k2l_ref, rtol=0, atol=1e-10)
    xo.assert_allclose(line['bi1.bsw1l1.3']._xobject.knl[2], -bsw_k2l_ref, rtol=0, atol=1e-10)
    xo.assert_allclose(line['bi1.bsw1l1.4']._xobject.knl[2], bsw_k2l_ref, rtol=0, atol=1e-10)
    xo.assert_allclose(line['bi1.bsw1l1.1'].k0, bsw_k0l_ref / line['bi1.bsw1l1.1'].length, rtol=0, atol=1e-10)
    xo.assert_allclose(line['bi1.bsw1l1.2'].k0, -bsw_k0l_ref / line['bi1.bsw1l1.2'].length, rtol=0, atol=1e-10)
    xo.assert_allclose(line['bi1.bsw1l1.3'].k0, -bsw_k0l_ref / line['bi1.bsw1l1.3'].length, rtol=0, atol=1e-10)
    xo.assert_allclose(line['bi1.bsw1l1.4'].k0, bsw_k0l_ref / line['bi1.bsw1l1.4'].length, rtol=0, atol=1e-10)

    tw = line.twiss()
    xo.assert_allclose(tw['x', 'bi1.tstr1l1'], -0.045739596, rtol=0, atol=1e-5)
    xo.assert_allclose(tw['y', 'bi1.tstr1l1'], 0.0000000, rtol=0, atol=1e-5)
    xo.assert_allclose(tw['betx', 'bi1.tstr1l1'], 5.201734, rtol=0, atol=1e-4)
    xo.assert_allclose(tw['bety', 'bi1.tstr1l1'], 6.900674, rtol=0, atol=1e-4)
    xo.assert_allclose(tw.qy, 4.474426935973243, rtol=0, atol=1e-6) # verify that it does not change from one version to the other
    xo.assert_allclose(tw.qx, 4.396759378318215, rtol=0, atol=1e-6)
    xo.assert_allclose(tw.dqy, -8.624431, rtol=0, atol=1e-3)
    xo.assert_allclose(tw.dqx, -3.560640, rtol=0, atol=1e-3)

    line.vars['bsw_k2l'] = bsw_k2l_ref / 3
    xo.assert_allclose(line['bi1.bsw1l1.1']._xobject.knl[2], bsw_k2l_ref / 3, rtol=0, atol=1e-10)
    xo.assert_allclose(line['bi1.bsw1l1.2']._xobject.knl[2], -bsw_k2l_ref / 3, rtol=0, atol=1e-10)
    xo.assert_allclose(line['bi1.bsw1l1.3']._xobject.knl[2], -bsw_k2l_ref / 3, rtol=0, atol=1e-10)
    xo.assert_allclose(line['bi1.bsw1l1.4']._xobject.knl[2], bsw_k2l_ref / 3, rtol=0, atol=1e-10)
    xo.assert_allclose(line['bi1.bsw1l1.1'].k0, bsw_k0l_ref / line['bi1.bsw1l1.1'].length, rtol=0, atol=1e-10)
    xo.assert_allclose(line['bi1.bsw1l1.2'].k0, -bsw_k0l_ref / line['bi1.bsw1l1.2'].length, rtol=0, atol=1e-10)
    xo.assert_allclose(line['bi1.bsw1l1.3'].k0, -bsw_k0l_ref / line['bi1.bsw1l1.3'].length, rtol=0, atol=1e-10)
    xo.assert_allclose(line['bi1.bsw1l1.4'].k0, bsw_k0l_ref / line['bi1.bsw1l1.4'].length, rtol=0, atol=1e-10)

    tw = line.twiss()
    xo.assert_allclose(tw['x', 'bi1.tstr1l1'], -0.045887, rtol=0, atol=1e-5)
    xo.assert_allclose(tw['y', 'bi1.tstr1l1'], 0.0000000, rtol=0, atol=1e-5)
    xo.assert_allclose(tw['betx', 'bi1.tstr1l1'], 5.264522, rtol=0, atol=1e-4)
    xo.assert_allclose(tw['bety', 'bi1.tstr1l1'], 6.317935, rtol=0, atol=1e-4)
    xo.assert_allclose(tw.qy, 4.4717791778, rtol=0, atol=1e-6)
    xo.assert_allclose(tw.qx, 4.3989420079, rtol=0, atol=1e-6)
    xo.assert_allclose(tw.dqy, -8.2045492725, rtol=0, atol=1e-3)
    xo.assert_allclose(tw.dqx, -3.5636322837, rtol=0, atol=1e-3)

    # Switch off bsws
    line.vars['bsw_k0l'] = 0
    line.vars['bsw_k2l'] = 0
    xo.assert_allclose(line['bi1.bsw1l1.1']._xobject.knl[2], 0, rtol=0, atol=1e-10)
    xo.assert_allclose(line['bi1.bsw1l1.2']._xobject.knl[2], 0, rtol=0, atol=1e-10)
    xo.assert_allclose(line['bi1.bsw1l1.3']._xobject.knl[2], 0, rtol=0, atol=1e-10)
    xo.assert_allclose(line['bi1.bsw1l1.4']._xobject.knl[2], 0, rtol=0, atol=1e-10)
    xo.assert_allclose(line['bi1.bsw1l1.1'].k0, 0, rtol=0, atol=1e-10)
    xo.assert_allclose(line['bi1.bsw1l1.2'].k0, 0, rtol=0, atol=1e-10)
    xo.assert_allclose(line['bi1.bsw1l1.3'].k0, 0, rtol=0, atol=1e-10)
    xo.assert_allclose(line['bi1.bsw1l1.4'].k0, 0, rtol=0, atol=1e-10)

    tw = line.twiss()
    xo.assert_allclose(tw['x', 'bi1.tstr1l1'], 0, rtol=0, atol=1e-5)
    xo.assert_allclose(tw['y', 'bi1.tstr1l1'], 0, rtol=0, atol=1e-5)
    xo.assert_allclose(tw['betx', 'bi1.tstr1l1'], 5.2996347, rtol=0, atol=1e-4)
    xo.assert_allclose(tw['bety', 'bi1.tstr1l1'], 3.838857, rtol=0, atol=1e-4)
    xo.assert_allclose(tw.qy, 4.45, rtol=0, atol=1e-6)
    xo.assert_allclose(tw.qx, 4.4, rtol=0, atol=1e-6)
    xo.assert_allclose(tw.dqy, -7.14978134184, rtol=0, atol=1e-3)
    xo.assert_allclose(tw.dqx, -3.56557575115, rtol=0, atol=1e-3)


    # Setup time-dependent functions

    df = pd.read_csv(test_data_folder /
                     'psb_chicane/chicane_collapse.csv',
                     delimiter=',', skipinitialspace=True)

    line.functions['fun_bsw_k0l'] = xd.FunctionPieceWiseLinear(
        x=df['time'].values, y=df['bsw_k0l'].values)
    line.functions['fun_bsw_k2l'] = xd.FunctionPieceWiseLinear(
        x=df['time'].values, y=df['bsw_k2l'].values)

    # Control knob with function
    line.vars['on_chicane_k0'] = 1
    line.vars['on_chicane_k2'] = 1
    line.vars['bsw_k0l'] = (line.functions.fun_bsw_k0l(line.vars['t_turn_s'])
                            * line.vars['on_chicane_k0'])
    line.vars['bsw_k2l'] = (line.functions.fun_bsw_k2l(line.vars['t_turn_s'])
                            * line.vars['on_chicane_k2'])


    # Test to_dict roundtrip
    line = xt.Line.from_dict(line.to_dict())
    line.build_tracker(_context=test_context)

    # Correct tunes and beta beat
    line.discard_tracker()
    line.insert_element(element=xt.Marker(), name='mker_match', at_s=79.874)
    line.build_tracker(_context=test_context)

    line.vars['on_chicane_k0'] = 0
    line.vars['on_chicane_k2'] = 0
    tw0 = line.twiss()
    line.vars['on_chicane_k0'] = 1
    line.vars['on_chicane_k2'] = 1


    t_correct = np.linspace(0, 5.5e-3, 30)

    kbrqf_corr_list = []
    kbrqd_corr_list = []
    kbrqd3corr_list = []
    kbrqd14corr_list = []
    for ii, tt in enumerate(t_correct):
        print(f'Correct tune at t = {tt * 1e3:.2f} ms   \n')
        line.vars['t_turn_s'] = tt

        line.match(
            #verbose=True,
            vary=[
                xt.Vary('kbrqfcorr', step=1e-4),
                xt.Vary('kbrqdcorr', step=1e-4),
                xt.Vary('kbrqd3corr', step=1e-4),
                xt.Vary('kbrqd14corr', step=1e-4),
            ],
            targets = [
                xt.Target('qx', value=tw0.qx, tol=1e-5, scale=1),
                xt.Target('qy', value=tw0.qy, tol=1e-5, scale=1),
                xt.Target('bety', at='mker_match',
                        value=tw0['bety', 'mker_match'], tol=1e-4, scale=100),
                xt.Target('alfy', at='mker_match',
                        value=tw0['alfy', 'mker_match'], tol=1e-4, scale=100)
            ]
        )

        kbrqf_corr_list.append(line.vars['kbrqfcorr']._value)
        kbrqd_corr_list.append(line.vars['kbrqdcorr']._value)
        kbrqd3corr_list.append(line.vars['kbrqd3corr']._value)
        kbrqd14corr_list.append(line.vars['kbrqd14corr']._value)

    line.functions['fun_kqf_corr'] = xd.FunctionPieceWiseLinear(
        x=t_correct, y=kbrqf_corr_list)
    line.functions['fun_kqd_corr'] = xd.FunctionPieceWiseLinear(
        x=t_correct, y=kbrqd_corr_list)
    line.functions['fun_qd3_corr'] = xd.FunctionPieceWiseLinear(
        x=t_correct, y=kbrqd3corr_list)
    line.functions['fun_qd14_corr'] = xd.FunctionPieceWiseLinear(
        x=t_correct, y=kbrqd14corr_list)

    line.vars['on_chicane_tune_corr'] = 1
    line.vars['kbrqfcorr'] = (line.vars['on_chicane_tune_corr']
                                * line.functions.fun_kqf_corr(line.vars['t_turn_s']))
    line.vars['kbrqdcorr'] = (line.vars['on_chicane_tune_corr']
                                * line.functions.fun_kqd_corr(line.vars['t_turn_s']))

    line.vars['on_chicane_beta_corr'] = 1
    line.vars['kbrqd3corr'] = (line.vars['on_chicane_beta_corr']
                            * line.functions.fun_qd3_corr(line.vars['t_turn_s']))
    line.vars['kbrqd14corr'] = (line.vars['on_chicane_beta_corr']
                            * line.functions.fun_qd14_corr(line.vars['t_turn_s']))

    line.vars['on_chicane_k0'] = 1
    line.vars['on_chicane_k2'] = 1
    line.vars['on_chicane_beta_corr'] = 0
    line.vars['on_chicane_tune_corr'] = 0

    line_thick = line.copy()
    line_thick.build_tracker(_context=test_context)

    line.discard_tracker()

    slicing_strategies = [
        Strategy(slicing=Teapot(1)),  # Default
        Strategy(slicing=Teapot(2), element_type=xt.Bend),
        Strategy(slicing=Teapot(2), element_type=xt.RBend),
        Strategy(slicing=Teapot(8), element_type=xt.Quadrupole),
    ]

    print("Slicing thick elements...")
    line.slice_thick_elements(slicing_strategies)

    line.build_tracker(_context=test_context)

    tw_thin = line.twiss()
    tw_thick = line_thick.twiss()

    print('\n')
    print(f'Qx: thick {tw_thin.qx:.4f} thin {tw_thick.qx:.4f}, diff {tw_thin.qx-tw_thick.qx:.4e}')
    print(f'Qy: thick {tw_thin.qy:.4f} thin {tw_thick.qy:.4f}, diff {tw_thin.qy-tw_thick.qy:.4e}')
    print(f"Q'x: thick {tw_thin.dqx:.4f} thin {tw_thick.dqx:.4f}, diff {tw_thin.dqx-tw_thick.dqx:.4f}")
    print(f"Q'y: thick {tw_thin.dqy:.4f} thin {tw_thick.dqy:.4f}, diff {tw_thin.dqy-tw_thick.dqy:.4f}")

    bety_interp = np.interp(tw_thick.s, tw_thin.s, tw_thin.bety)
    print(f"Max beta beat: {np.max(np.abs(tw_thick.bety/bety_interp - 1)):.4e}")

    #####################
    # Check against ptc #
    #####################

    with open(test_data_folder / 'psb_chicane/ptc_ref.json', 'r') as fid: # generated by xtrack/examples/psb/001_compare_against_ptc.py
        ptc_ref = json.load(fid)

    for kk, vv in ptc_ref.items():
        ptc_ref[kk] = np.array(vv)

    t_test = ptc_ref['t_test']
    qx_ptc = ptc_ref['qx_ptc']
    qy_ptc = ptc_ref['qy_ptc']
    dqx_ptc = ptc_ref['dqx_ptc']
    dqy_ptc = ptc_ref['dqy_ptc']
    bety_at_scraper_ptc = ptc_ref['bety_at_scraper_ptc']

    # Check against ptc (correction off)
    line.vars['on_chicane_beta_corr'] = 0
    line.vars['on_chicane_tune_corr'] = 0
    qx_thick = []
    qy_thick = []
    dqx_thick = []
    dqy_thick = []
    bety_at_scraper_thick = []
    qx_thin = []
    qy_thin = []
    dqx_thin = []
    dqy_thin = []
    bety_at_scraper_thin = []
    for ii, tt in enumerate(t_test):
        print(f'Check against ptc, twiss at t = {tt*1e3:.2f} ms   ', end='\r', flush=True)
        line_thick.vars['t_turn_s'] = tt
        line.vars['t_turn_s'] = tt

        tw_thick = line_thick.twiss()
        bety_at_scraper_thick.append(tw_thick['bety', 'br.stscrap22'])
        qx_thick.append(tw_thick.qx)
        qy_thick.append(tw_thick.qy)
        dqx_thick.append(tw_thick.dqx)
        dqy_thick.append(tw_thick.dqy)

        tw_thin = line.twiss()
        bety_at_scraper_thin.append(tw_thin['bety', 'br.stscrap22'])
        qx_thin.append(tw_thin.qx)
        qy_thin.append(tw_thin.qy)
        dqx_thin.append(tw_thin.dqx)
        dqy_thin.append(tw_thin.dqy)


    xo.assert_allclose(qx_thick, qx_ptc, atol=2e-4, rtol=0)
    xo.assert_allclose(qy_thick, qy_ptc, atol=2e-4, rtol=0)
    xo.assert_allclose(qx_thin, qx_ptc, atol=1e-3, rtol=0)
    xo.assert_allclose(qy_thin, qy_ptc, atol=1e-3, rtol=0)
    xo.assert_allclose(dqx_thick, dqx_ptc, atol=0.5, rtol=0)
    xo.assert_allclose(dqy_thick, dqy_ptc, atol=0.5, rtol=0)
    xo.assert_allclose(dqx_thin, dqx_ptc, atol=0.5, rtol=0)
    xo.assert_allclose(dqy_thin, dqy_ptc, atol=0.5, rtol=0)
    xo.assert_allclose(bety_at_scraper_thick, bety_at_scraper_ptc, atol=0, rtol=1e-2)
    xo.assert_allclose(bety_at_scraper_thin, bety_at_scraper_ptc, atol=0, rtol=2e-2)

    # Check correction
    line.vars['on_chicane_beta_corr'] = 1
    line.vars['on_chicane_tune_corr'] = 1
    line_thick.vars['on_chicane_beta_corr'] = 1
    line_thick.vars['on_chicane_tune_corr'] = 1
    qx_thick_corr = []
    qy_thick_corr = []
    bety_at_scraper_thick_corr = []
    qx_thin_corr = []
    qy_thin_corr = []
    bety_at_scraper_thin_corr = []
    for ii, tt in enumerate(t_test):
        print(f'Check correction, twiss at t = {tt*1e3:.2f} ms   ', end='\r', flush=True)
        line_thick.vars['t_turn_s'] = tt
        line.vars['t_turn_s'] = tt

        tw_thick = line_thick.twiss()
        bety_at_scraper_thick_corr.append(tw_thick['bety', 'br.stscrap22'])
        qx_thick_corr.append(tw_thick.qx)
        qy_thick_corr.append(tw_thick.qy)

        tw_thin = line.twiss()
        bety_at_scraper_thin_corr.append(tw_thin['bety', 'br.stscrap22'])
        qx_thin_corr.append(tw_thin.qx)
        qy_thin_corr.append(tw_thin.qy)

    qx_thick_corr = np.array(qx_thick_corr)
    qy_thick_corr = np.array(qy_thick_corr)
    bety_at_scraper_thick_corr = np.array(bety_at_scraper_thick_corr)
    qx_thin_corr = np.array(qx_thin_corr)
    qy_thin_corr = np.array(qy_thin_corr)
    bety_at_scraper_thin_corr = np.array(bety_at_scraper_thin_corr)

    xo.assert_allclose(qx_thick_corr, qx_ptc[-1], atol=3e-3, rtol=0)
    xo.assert_allclose(qy_thick_corr, qy_ptc[-1], atol=3e-3, rtol=0)
    xo.assert_allclose(qx_thin_corr, qx_ptc[-1], atol=3e-3, rtol=0)
    xo.assert_allclose(qy_thin_corr, qy_ptc[-1], atol=3e-3, rtol=0)
    xo.assert_allclose(bety_at_scraper_thick_corr, bety_at_scraper_ptc[-1], atol=0, rtol=3e-2)
    xo.assert_allclose(bety_at_scraper_thin_corr, bety_at_scraper_ptc[-1], atol=0, rtol=3e-2)

    # Tracking with time-dependent variable

    line.vars['on_chicane_k0'] = 1
    line.vars['on_chicane_k2'] = 1
    line.vars['on_chicane_beta_corr'] = 1
    line.vars['on_chicane_tune_corr'] = 1

    # Install monitor at foil
    monitor = xt.ParticlesMonitor(start_at_turn=0, stop_at_turn=6000, num_particles=1)
    line.discard_tracker()
    line.insert_element(index='bi1.tstr1l1', element=monitor, name='monitor_at_foil')
    line.build_tracker(_context=test_context)

    p = line.build_particles(x=0, px=0, y=0, py=0, delta=0, zeta=0)

    line.enable_time_dependent_vars = True
    line.dt_update_time_dependent_vars = 3e-6

    print('Tracking...')
    line.track(p, num_turns=6000, time=True)
    print(f'Done in {line.time_last_track:.4} s')

    xo.assert_allclose(monitor.x[0, 0], -0.045936, rtol=0, atol=1e-5)
    xo.assert_allclose(monitor.x[0, 300], -0.04522354, rtol=0, atol=1e-5)
    xo.assert_allclose(monitor.x[0, 2500], -0.02256763, rtol=0, atol=1e-5)
    xo.assert_allclose(monitor.x[0, 4500], -0.00143883, rtol=0, atol=1e-5)
    xo.assert_allclose(monitor.x[0, 5500], 0.0, rtol=0, atol=1e-5)

    # Test multiturn injection
    if isinstance(test_context, xo.ContextCpu):
        line.t_turn_s = 0 # Reset time!

        line.vars['on_chicane_k0'] = 1
        line.vars['on_chicane_k2'] = 1
        line.vars['on_chicane_beta_corr'] = 1
        line.vars['on_chicane_tune_corr'] = 1

        df = pd.read_table(test_data_folder / 'psb_chicane/inj_distrib.dat',
            skiprows=3,
            names="x x' y y' z z' Phase Time Energy Loss".split())

        kin_energy_ev = df.Energy.values * 1e6
        tot_energy_ev = kin_energy_ev + xt.PROTON_MASS_EV
        p0c = line.particle_ref.p0c[0]
        tot_energy0_ev = line.particle_ref.energy0[0]
        ptau = (tot_energy_ev - tot_energy0_ev) / p0c

        part_for_injection = xt.Particles(q0=1, mass0=xt.PROTON_MASS_EV, p0c=line.particle_ref.p0c[0],
                                        ptau=ptau)

        part_for_injection.x = df.x.values * 1e-3
        part_for_injection.y = df.y.values * 1e-3
        part_for_injection.zeta = df.z.values * 1e-3
        part_for_injection.px = df["x'"].values  * 1e-3 * (1 + part_for_injection.delta)
        part_for_injection.py = df["y'"].values  * 1e-3 * (1 + part_for_injection.delta)
        part_for_injection.weight = 10

        p_injection = xt.ParticlesInjectionSample(particles_to_inject=part_for_injection,
                                                line=line,
                                                element_name='injection',
                                                num_particles_to_inject=7)

        line.discard_tracker()
        line.insert_element(index='bi1.tstr1l1', element=p_injection, name='injection')

        # Add monitor at end line
        monitor = xt.ParticlesMonitor(start_at_turn=0, stop_at_turn=10, num_particles=100)
        line.insert_element(index='p16ring1$end', element=monitor, name='monitor_at_end')
        line.build_tracker()

        p = line.build_particles(_capacity=100, x=0)
        p.state[0] = -500 # kill the particle added by default

        intensity = []
        line.enable_time_dependent_vars = True
        for iturn in range(8):
            intensity.append(p.weight[p.state>0].sum())
            line.track(p, num_turns=1)

        assert np.all(np.sum(monitor.state > 0, axis=0)
                    == np.array([ 0,  7, 14, 21, 28, 35, 42, 49, 56,  0]))

        assert np.all(monitor.at_element[monitor.state > 0] ==
                    line.element_names.index('monitor_at_end'))

        xo.assert_allclose(monitor.s[monitor.state > 0],
                        line.get_s_position('monitor_at_end'), atol=1e-12)
