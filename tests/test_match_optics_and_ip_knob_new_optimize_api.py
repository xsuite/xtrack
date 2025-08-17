import pathlib

import numpy as np

import xobjects as xo
import xtrack as xt
from xobjects.test_helpers import for_all_test_contexts

test_data_folder = pathlib.Path(
    __file__).parent.joinpath('../test_data').absolute()

@for_all_test_contexts
def test_ip_knob_matching_new_optimize_api(test_context):

    collider = xt.load(test_data_folder /
                    'hllhc15_thick/hllhc15_collider_thick.json')
    collider.build_trackers(test_context)

    # Add default steps and limits

    limitmcbxh =  63.5988e-6
    limitmcbxv =  67.0164e-6
    limitmcbx  =  67.0164e-6
    limitmcby  =  96.3000e-6
    limitmcb   =  80.8000e-6
    limitmcbc  =  89.8700e-6
    limitmcbw  =  80.1400e-6
    nrj = 7000.
    scale = 23348.89927*0.9
    scmin = 0.03*7000./nrj

    collider.vars.vary_default.update({
        # Correctors in IR8 (Q4 and Q5, and triplet)
        'acbyhs4.l8b1':{'step': 1.0e-15,'limits':(-limitmcby, limitmcby)},
        'acbyhs4.r8b2':{'step': 1.0e-15,'limits':(-limitmcby, limitmcby)},
        'acbyhs4.l8b2':{'step': 1.0e-15,'limits':(-limitmcby, limitmcby)},
        'acbyhs4.r8b1':{'step': 1.0e-15,'limits':(-limitmcby, limitmcby)},
        'acbchs5.l8b2':{'step': 1.0e-15,'limits':(-limitmcby, limitmcby)},
        'acbchs5.l8b1':{'step': 1.0e-15,'limits':(-limitmcby, limitmcby)},
        'acbyhs5.r8b1':{'step': 1.0e-15,'limits':(-limitmcbc, limitmcbc)},
        'acbyhs5.r8b2':{'step': 1.0e-15,'limits':(-limitmcbc, limitmcbc)},
        'acbxh1.l8':   {'step': 1.0e-15, 'limits':(-limitmcbx, limitmcbx)},
        'acbxh2.l8':   {'step': 1.0e-15, 'limits':(-limitmcbx, limitmcbx)},
        'acbxh3.l8':   {'step': 1.0e-15, 'limits':(-limitmcbx, limitmcbx)},
        'acbxh1.r8':   {'step': 1.0e-15, 'limits':(-limitmcbx, limitmcbx)},
        'acbxh2.r8':   {'step': 1.0e-15, 'limits':(-limitmcbx, limitmcbx)},
        'acbxh3.r8':   {'step': 1.0e-15, 'limits':(-limitmcbx, limitmcbx)},
    })

    # Check a few steps and limits

    xo.assert_allclose(collider.vars.vary_default['acbxh1.l8']['step'], 1.0e-15, atol=1e-17, rtol=0)
    xo.assert_allclose(collider.vars.vary_default['acbxh1.l8']['limits'][0], -limitmcbx, atol=1e-17, rtol=0)
    xo.assert_allclose(collider.vars.vary_default['acbxh1.l8']['limits'][1], limitmcbx, atol=1e-17, rtol=0)
    xo.assert_allclose(collider.vars.vary_default['acbyhs5.r8b2']['step'], 1.0e-15, atol=1e-17, rtol=0)
    xo.assert_allclose(collider.vars.vary_default['acbyhs5.r8b2']['limits'][0], -limitmcbc, atol=1e-17, rtol=0)
    xo.assert_allclose(collider.vars.vary_default['acbyhs5.r8b2']['limits'][1], limitmcbc, atol=1e-17, rtol=0)

    # Check that they are preserved by to_dict/from_dict
    collider = xt.Environment.from_dict(collider.to_dict())
    collider.build_trackers()

    xo.assert_allclose(collider.vars.vary_default['acbxh1.l8']['step'], 1.0e-15, atol=1e-17, rtol=0)
    xo.assert_allclose(collider.vars.vary_default['acbxh1.l8']['limits'][0], -limitmcbx, atol=1e-17, rtol=0)
    xo.assert_allclose(collider.vars.vary_default['acbxh1.l8']['limits'][1], limitmcbx, atol=1e-17, rtol=0)
    xo.assert_allclose(collider.vars.vary_default['acbyhs5.r8b2']['step'], 1.0e-15, atol=1e-17, rtol=0)
    xo.assert_allclose(collider.vars.vary_default['acbyhs5.r8b2']['limits'][0], -limitmcbc, atol=1e-17, rtol=0)
    xo.assert_allclose(collider.vars.vary_default['acbyhs5.r8b2']['limits'][1], limitmcbc, atol=1e-17, rtol=0)

    # kill all existing knobs orbit knobs in ip2 and ip8
    all_knobs_ip2ip8 = [
        'acbxh3.r2', 'acbchs5.r2b1', 'pxip2b1', 'acbxh2.l8',
        'acbyhs4.r8b2', 'pyip2b1', 'acbxv1.l8', 'acbyvs4.l2b1', 'acbxh1.l8',
        'acbxv2.r8', 'pxip8b2', 'yip8b1', 'pxip2b2', 'acbcvs5.r2b1', 'acbyhs4.l8b1',
        'acbyvs4.l8b1', 'acbxh2.l2', 'acbxh3.l2', 'acbxv1.r8', 'acbxv1.r2',
        'acbyvs4.r2b2', 'acbyvs4.l2b2', 'yip8b2', 'xip2b2', 'acbxh2.r2',
        'acbyhs4.l2b2', 'acbxv2.r2', 'acbyhs5.r8b1', 'acbxh2.r8', 'acbxv3.r8',
        'acbyvs5.r8b2', 'acbyvs5.l2b2', 'yip2b1', 'acbxv2.l2', 'acbyhs4.r2b2',
        'acbyhs4.r2b1', 'xip8b2', 'acbyvs5.l2b1', 'acbyvs4.r8b1', 'acbyvs4.r8b2',
        'acbyvs5.r8b1', 'acbxh1.r8', 'acbyvs4.l8b2', 'acbyhs5.l2b1', 'acbyvs4.r2b1',
        'acbcvs5.r2b2', 'acbcvs5.l8b2', 'acbyhs4.r8b1', 'pxip8b1', 'acbxv1.l2',
        'yip2b2', 'acbyhs4.l8b2', 'acbxv3.r2', 'xip8b1', 'acbchs5.r2b2', 'acbxh3.l8',
        'acbxh3.r8', 'acbyhs5.r8b2', 'acbxv2.l8', 'acbxh1.l2', 'pyip8b1', 'pyip8b2',
        'acbxv3.l8', 'xip2b1', 'acbyhs5.l2b2', 'acbchs5.l8b2', 'acbcvs5.l8b1',
        'pyip2b2', 'acbxv3.l2', 'acbchs5.l8b1', 'acbyhs4.l2b1', 'acbxh1.r2']

    for kk in all_knobs_ip2ip8:
        collider.vars[kk] = 0

    # Match horizontal xing angle in ip8
    angle_match = 300e-6
    opt = collider.match_knob(
        run=False,
        default_tol={'x': 1.1e-10, 'px': 0.9e-10},
        knob_name='on_x8h',
        knob_value_start=0,
        knob_value_end=(angle_match * 1e6),
        start=['s.ds.l8.b1', 's.ds.l8.b2'],
        end=['e.ds.r8.b1', 'e.ds.r8.b2'],
        init=[xt.TwissInit(), xt.TwissInit()],
        targets=[
            xt.TargetSet(x=0, px=0, at=xt.END, line='lhcb1'),
            xt.TargetSet(x=0, px=0, at=xt.END, line='lhcb2'),
            xt.Target('x', 0, at='ip8', line='lhcb1'),
            xt.Target('px', angle_match, at='ip8', line='lhcb1', tol=0.8e-10),
            xt.TargetSet(x=0, px=-angle_match, at='ip8', line='lhcb2'),
        ],
        vary=[
            # Vary with custom step and limits
            xt.VaryList(['acbyhs4.l8b1'], step=2e-15, limits=(-9e-5, 9e-5)),
            # Vary using default step and limits
            xt.VaryList([
                'acbyhs4.r8b2', 'acbyhs4.l8b2', 'acbyhs4.r8b1',
                'acbchs5.l8b2', 'acbchs5.l8b1', 'acbyhs5.r8b1', 'acbyhs5.r8b2']),
            xt.VaryList([
                'acbxh1.l8', 'acbxh2.l8', 'acbxh3.l8',
                'acbxh1.r8', 'acbxh2.r8', 'acbxh3.r8'], tag='mcbx')]
        )

    ll = opt.log()
    assert len(ll) == 1
    assert ll.iteration[0] == 0
    xo.assert_allclose(ll['penalty', 0], 0.0424264, atol=1e-6, rtol=0)
    assert ll['tol_met', 0] == 'yyyyynyn'
    assert ll['target_active', 0] == 'yyyyyyyy'
    assert ll['vary_active', 0] == 'yyyyyyyyyyyyyy'

    vnames = [vv.name for vv in opt.vary]
    assert np.all(np.array(vnames) == np.array(
    ['acbyhs4.l8b1_from_on_x8h', 'acbyhs4.r8b2_from_on_x8h',
    'acbyhs4.l8b2_from_on_x8h', 'acbyhs4.r8b1_from_on_x8h',
    'acbchs5.l8b2_from_on_x8h', 'acbchs5.l8b1_from_on_x8h',
    'acbyhs5.r8b1_from_on_x8h', 'acbyhs5.r8b2_from_on_x8h',
    'acbxh1.l8_from_on_x8h', 'acbxh2.l8_from_on_x8h', 'acbxh3.l8_from_on_x8h',
    'acbxh1.r8_from_on_x8h', 'acbxh2.r8_from_on_x8h', 'acbxh3.r8_from_on_x8h']))

    vtags = [vv.tag for vv in opt.vary]
    assert np.all(np.array(vtags) == np.array(
        ['', '', '', '', '', '', '', '', 'mcbx', 'mcbx', 'mcbx', 'mcbx', 'mcbx', 'mcbx']))

    xo.assert_allclose(opt.vary[0].step, 2e-15, atol=1e-17, rtol=0)
    xo.assert_allclose(opt.vary[0].limits[0], -9e-5, atol=1e-10, rtol=0)
    xo.assert_allclose(opt.vary[0].limits[1], 9e-5, atol=1e-10, rtol=0)
    xo.assert_allclose(opt.vary[2].step, 1e-15, atol=1e-17, rtol=0)
    xo.assert_allclose(opt.vary[2].limits[0], -limitmcby, atol=1e-10, rtol=0)
    xo.assert_allclose(opt.vary[2].limits[1], limitmcby, atol=1e-10, rtol=0)

    xo.assert_allclose(opt.targets[4].tol, 1.1e-10, atol=1e-14, rtol=0)
    xo.assert_allclose(opt.targets[5].tol, 0.8e-10, atol=1e-14, rtol=0)
    xo.assert_allclose(opt.targets[6].tol, 1.1e-10, atol=1e-14, rtol=0)
    xo.assert_allclose(opt.targets[7].tol, 0.9e-10, atol=1e-14, rtol=0)

    # Set mcmbx by hand (as in mad-x script)
    testkqx8=abs(collider.varval['kqx.l8'])*7000./0.3
    if testkqx8> 210.:
        acbx_xing_ir8 = 1.0e-6   # Value for 170 urad crossing
    else:
        acbx_xing_ir8 = 11.0e-6  # Value for 170 urad crossing

    # # As in mad-x script
    # collider.vars['acbxh1.l8_from_on_x8h'] = acbx_xing_ir8 * angle_match / 170e-6
    # collider.vars['acbxh2.l8_from_on_x8h'] = acbx_xing_ir8 * angle_match / 170e-6
    # collider.vars['acbxh3.l8_from_on_x8h'] = acbx_xing_ir8 * angle_match / 170e-6
    # collider.vars['acbxh1.r8_from_on_x8h'] = -acbx_xing_ir8 * angle_match / 170e-6
    # collider.vars['acbxh2.r8_from_on_x8h'] = -acbx_xing_ir8 * angle_match / 170e-6
    # collider.vars['acbxh3.r8_from_on_x8h'] = -acbx_xing_ir8 * angle_match / 170e-6

    # Harder case to check mcbx matching
    collider.vars['acbxh1.l8_from_on_x8h'] = acbx_xing_ir8 * angle_match / 170e-6 * 0.1
    collider.vars['acbxh2.l8_from_on_x8h'] = acbx_xing_ir8 * angle_match / 170e-6 * 0.1
    collider.vars['acbxh3.l8_from_on_x8h'] = acbx_xing_ir8 * angle_match / 170e-6 * 0.1
    collider.vars['acbxh1.r8_from_on_x8h'] = -acbx_xing_ir8 * angle_match / 170e-6 * 0.1
    collider.vars['acbxh2.r8_from_on_x8h'] = -acbx_xing_ir8 * angle_match / 170e-6 * 0.1
    collider.vars['acbxh3.r8_from_on_x8h'] = -acbx_xing_ir8 * angle_match / 170e-6 * 0.1

    init_mcbx_plus = collider.varval['acbxh1.l8_from_on_x8h']

    # First round of optimization without changing mcbx
    opt.disable(vary='mcbx')
    vtags = [vv.tag for vv in opt.vary]
    assert np.all(np.array(vtags) == np.array(
        ['', '', '', '', '', '', '', '', 'mcbx', 'mcbx', 'mcbx', 'mcbx', 'mcbx', 'mcbx']))
    vactive = [vv.active for vv in opt.vary]
    assert np.all(np.array(vactive) == np.array(
        [True, True, True, True, True, True, True, True, False, False, False, False, False, False]))

    opt.step(10) # perform 10 steps without checking for convergence

    ll = opt.log()
    assert 11 <= len(ll) <= 13
    assert ll['vary_active', 0] == 'yyyyyyyyyyyyyy'
    assert ll['vary_active', 1] == 'yyyyyyyynnnnnn'
    assert ll['vary_active', len(ll) - 1] == 'yyyyyyyynnnnnn'

    # Check solution not found
    assert ll['tol_met', len(ll) - 1] != 'yyyyyyyy'

    # Check that mcbxs did not move
    xo.assert_allclose(ll['vary_8', 1:], init_mcbx_plus, atol=1e-12, rtol=0)
    xo.assert_allclose(ll['vary_9', 1:], init_mcbx_plus, atol=1e-12, rtol=0)
    xo.assert_allclose(ll['vary_10', 1:], init_mcbx_plus, atol=1e-12, rtol=0)
    xo.assert_allclose(ll['vary_11', 1:], -init_mcbx_plus, atol=1e-12, rtol=0)
    xo.assert_allclose(ll['vary_12', 1:], -init_mcbx_plus, atol=1e-12, rtol=0)
    xo.assert_allclose(ll['vary_13', 1:], -init_mcbx_plus, atol=1e-12, rtol=0)


    # Link all mcbx stengths to the first one
    collider.vars['acbxh2.l8_from_on_x8h'] =  collider.vars['acbxh1.l8_from_on_x8h']
    collider.vars['acbxh3.l8_from_on_x8h'] =  collider.vars['acbxh1.l8_from_on_x8h']
    collider.vars['acbxh2.r8_from_on_x8h'] = -collider.vars['acbxh1.l8_from_on_x8h']
    collider.vars['acbxh3.r8_from_on_x8h'] = -collider.vars['acbxh1.l8_from_on_x8h']
    collider.vars['acbxh1.r8_from_on_x8h'] = -collider.vars['acbxh1.l8_from_on_x8h']

    # Enable first mcbx knob (which controls the others)
    assert opt.vary[8].name == 'acbxh1.l8_from_on_x8h'
    opt.vary[8].active = True

    opt.solve()
    ll = opt.log()

    # Check driving knob is enabled
    assert np.all(ll['vary_active', 13:] == 'yyyyyyyyynnnnn')

    # Check solution found
    assert np.all(ll['tol_met', -1] == 'yyyyyyyy')

    # Check imposed relationship among varys
    xo.assert_allclose(ll['vary_8', 13], ll['vary_9',   13], atol=1e-12, rtol=0)
    xo.assert_allclose(ll['vary_8', 13], ll['vary_10',  13], atol=1e-12, rtol=0)
    xo.assert_allclose(ll['vary_8', 13], -ll['vary_11', 13], atol=1e-12, rtol=0)
    xo.assert_allclose(ll['vary_8', 13], -ll['vary_12', 13], atol=1e-12, rtol=0)
    xo.assert_allclose(ll['vary_8', 13], -ll['vary_13', 13], atol=1e-12, rtol=0)

    opt.generate_knob()

    collider.vars['on_x8h'] = 100
    tw = collider.twiss()
    collider.vars['on_x8h'] = 0

    xo.assert_allclose(tw.lhcb1['x', 'ip8'], 0, atol=1e-10, rtol=0)
    xo.assert_allclose(tw.lhcb2['x', 'ip8'], 0, atol=1e-10, rtol=0)
    xo.assert_allclose(tw.lhcb1['px', 'ip8'], 100e-6, atol=1e-10, rtol=0)
    xo.assert_allclose(tw.lhcb2['px', 'ip8'], -100e-6, atol=1e-10, rtol=0)

    # Match horizontal separation in ip8
    sep_match = 2e-3
    opt = collider.match_knob(
        run=False,
        knob_name='on_sep8h',
        knob_value_start=0,
        knob_value_end=(sep_match * 1e3),
        start=['s.ds.l8.b1', 's.ds.l8.b2'],
        end=['e.ds.r8.b1', 'e.ds.r8.b2'],
        init=[xt.TwissInit(betx=1, bety=1, element_name='s.ds.l8.b1', line=collider.lhcb1),
                    xt.TwissInit(betx=1, bety=1, element_name='s.ds.l8.b2', line=collider.lhcb2)],
        targets=[
            xt.TargetList(['x', 'px'], at='e.ds.r8.b1', line='lhcb1', value=0),
            xt.TargetList(['x', 'px'], at='e.ds.r8.b2', line='lhcb2', value=0),
            xt.Target('x', sep_match, at='ip8', line='lhcb1'),
            xt.Target('x', -sep_match, at='ip8', line='lhcb2'),
            xt.Target('px', 0, at='ip8', line='lhcb1'),
            xt.Target('px', 0, at='ip8', line='lhcb2'),
        ],
        vary=[
            xt.VaryList([
                'acbyhs4.l8b1', 'acbyhs4.r8b2', 'acbyhs4.l8b2', 'acbyhs4.r8b1',
                'acbchs5.l8b2', 'acbchs5.l8b1', 'acbyhs5.r8b1', 'acbyhs5.r8b2']),
            xt.VaryList([
                'acbxh1.l8', 'acbxh2.l8', 'acbxh3.l8',
                'acbxh1.r8', 'acbxh2.r8', 'acbxh3.r8'], tag='mcbx')]
        )

    # Set mcmbx by hand (as in mad-x script)
    testkqx8 = abs(collider.varval['kqx.l8'])*7000./0.3
    if testkqx8 > 210.:
        acbx_sep_ir8 = 18e-6   # Value for 170 urad crossing
    else:
        acbx_sep_ir8 = 16e-6  # Value for 170 urad crossing

    collider.vars['acbxh1.l8_from_on_sep8h'] = acbx_sep_ir8 * sep_match / 2e-3
    collider.vars['acbxh2.l8_from_on_sep8h'] = acbx_sep_ir8 * sep_match / 2e-3
    collider.vars['acbxh3.l8_from_on_sep8h'] = acbx_sep_ir8 * sep_match / 2e-3
    collider.vars['acbxh1.r8_from_on_sep8h'] = acbx_sep_ir8 * sep_match / 2e-3
    collider.vars['acbxh2.r8_from_on_sep8h'] = acbx_sep_ir8 * sep_match / 2e-3
    collider.vars['acbxh3.r8_from_on_sep8h'] = acbx_sep_ir8 * sep_match / 2e-3

    # First round of optimization without changing mcbx
    opt.disable(vary='mcbx')
    opt.step(10) # perform 10 steps without checking for convergence

    # Enable first mcbx knob (which controls the others)
    assert opt.vary[8].name == 'acbxh1.l8_from_on_sep8h'
    opt.vary[8].active = True

    opt.solve()
    opt.generate_knob()

    collider.vars['on_sep8h'] = 1.5
    tw = collider.twiss()
    collider.vars['on_sep8h'] = 0

    xo.assert_allclose(tw.lhcb1['x', 'ip8'], 1.5e-3, atol=1e-9, rtol=0)
    xo.assert_allclose(tw.lhcb2['x', 'ip8'], -1.5e-3, atol=1e-9, rtol=0)
    xo.assert_allclose(tw.lhcb1['px', 'ip8'], 0, atol=1e-9, rtol=0)
    xo.assert_allclose(tw.lhcb2['px', 'ip8'], 0, atol=1e-9, rtol=0)

    # Check that on_x8h still works
    collider.vars['on_x8h'] = 100
    tw = collider.twiss()
    collider.vars['on_x8h'] = 0

    xo.assert_allclose(tw.lhcb1['x', 'ip8'], 0, atol=1e-9, rtol=0)
    xo.assert_allclose(tw.lhcb2['x', 'ip8'], 0, atol=1e-9, rtol=0)
    xo.assert_allclose(tw.lhcb1['px', 'ip8'], 100e-6, atol=1e-9, rtol=0)
    xo.assert_allclose(tw.lhcb2['px', 'ip8'], -100e-6, atol=1e-9, rtol=0)

    # Both knobs together
    collider.vars['on_x8h'] = 120
    collider.vars['on_sep8h'] = 1.7
    tw = collider.twiss()
    collider.vars['on_x8h'] = 0
    collider.vars['on_sep8h'] = 0

    xo.assert_allclose(tw.lhcb1['x', 'ip8'], 1.7e-3, atol=1e-9, rtol=0)
    xo.assert_allclose(tw.lhcb2['x', 'ip8'], -1.7e-3, atol=1e-9, rtol=0)
    xo.assert_allclose(tw.lhcb1['px', 'ip8'], 120e-6, atol=1e-9, rtol=0)
    xo.assert_allclose(tw.lhcb2['px', 'ip8'], -120e-6, atol=1e-9, rtol=0)

@for_all_test_contexts
def test_match_ir8_optics_new_optimize_api(test_context):

    collider = xt.load(test_data_folder /
                    'hllhc15_thick/hllhc15_collider_thick.json')
    collider.build_trackers(test_context)

    tw = collider.twiss()
    xo.assert_allclose(tw.lhcb1['betx', 'ip1'], 0.15, atol=1e-6, rtol=0)
    xo.assert_allclose(tw.lhcb1['bety', 'ip1'], 0.15, atol=1e-6, rtol=0)
    xo.assert_allclose(tw.lhcb2['betx', 'ip1'], 0.15, atol=1e-6, rtol=0)
    xo.assert_allclose(tw.lhcb2['bety', 'ip1'], 0.15, atol=1e-6, rtol=0)

    xo.assert_allclose(tw.lhcb1['betx', 'ip5'], 0.15, atol=1e-6, rtol=0)
    xo.assert_allclose(tw.lhcb1['bety', 'ip5'], 0.15, atol=1e-6, rtol=0)
    xo.assert_allclose(tw.lhcb2['betx', 'ip5'], 0.15, atol=1e-6, rtol=0)
    xo.assert_allclose(tw.lhcb2['bety', 'ip5'], 0.15, atol=1e-6, rtol=0)

    xo.assert_allclose(tw.lhcb1['betx', 'ip8'], 1.5, atol=1e-6, rtol=0)
    xo.assert_allclose(tw.lhcb1['bety', 'ip8'], 1.5, atol=1e-6, rtol=0)
    xo.assert_allclose(tw.lhcb2['betx', 'ip8'], 1.5, atol=1e-6, rtol=0)
    xo.assert_allclose(tw.lhcb2['bety', 'ip8'], 1.5, atol=1e-6, rtol=0)

    xo.assert_allclose(tw.lhcb1['betx', 'ip2'], 10., atol=1e-5, rtol=0)
    xo.assert_allclose(tw.lhcb1['bety', 'ip2'], 10., atol=1e-5, rtol=0)
    xo.assert_allclose(tw.lhcb2['betx', 'ip2'], 10., atol=1e-5, rtol=0)
    xo.assert_allclose(tw.lhcb2['bety', 'ip2'], 10., atol=1e-5, rtol=0)


    nrj = 7000.
    scale = 23348.89927*0.9
    scmin = 0.03*7000./nrj
    qtlimitx28 = 1.0*225.0/scale
    qtlimitx15 = 1.0*205.0/scale
    qtlimit2 = 1.0*160.0/scale
    qtlimit3 = 1.0*200.0/scale
    qtlimit4 = 1.0*125.0/scale
    qtlimit5 = 1.0*120.0/scale
    qtlimit6 = 1.0*90.0/scale

    collider.vars.vary_default.update({
        'kq4.r8b1':    {'step': 1.0E-6, 'limits': ( qtlimit2*scmin, qtlimit2)},
        'kq5.r8b1':    {'step': 1.0E-6, 'limits': (-qtlimit2, -qtlimit2*scmin)},
        'kq6.r8b1':    {'step': 1.0E-6, 'limits': ( qtlimit2*scmin, qtlimit2)},
        'kq7.r8b1':    {'step': 1.0E-6, 'limits': (-qtlimit3, -qtlimit3*scmin)},
        'kq8.r8b1':    {'step': 1.0E-6, 'limits': ( qtlimit3*scmin, qtlimit3)},
        'kq9.r8b1':    {'step': 1.0E-6, 'limits': (-qtlimit3, -qtlimit3*scmin)},
        'kq10.r8b1':   {'step': 1.0E-6, 'limits': ( qtlimit3*scmin, qtlimit3)},
        'kqtl11.r8b1': {'step': 1.0E-6, 'limits': (-qtlimit4, qtlimit4)},
        'kqt12.r8b1':  {'step': 1.0E-6, 'limits': (-qtlimit5, qtlimit5)},
        'kqt13.r8b1':  {'step': 1.0E-6, 'limits': (-qtlimit5, qtlimit5)},
        'kq4.l8b1':    {'step': 1.0E-6, 'limits': (-qtlimit2, -qtlimit2*scmin)},
        'kq5.l8b1':    {'step': 1.0E-6, 'limits': ( qtlimit2*scmin, qtlimit2)},
        'kq6.l8b1':    {'step': 1.0E-6, 'limits': (-qtlimit3, -qtlimit3*scmin)},
        'kq7.l8b1':    {'step': 1.0E-6, 'limits': ( qtlimit3*scmin, qtlimit3)},
        'kq8.l8b1':    {'step': 1.0E-6, 'limits': (-qtlimit3, -qtlimit3*scmin)},
        'kq9.l8b1':    {'step': 1.0E-6, 'limits': ( qtlimit3*scmin, qtlimit3)},
        'kq10.l8b1':   {'step': 1.0E-6, 'limits': (-qtlimit3, -qtlimit3*scmin)},
        'kqtl11.l8b1': {'step': 1.0E-6, 'limits': (-qtlimit4, qtlimit4)},
        'kqt12.l8b1':  {'step': 1.0E-6, 'limits': (-qtlimit5, qtlimit5)},
        'kqt13.l8b1':  {'step': 1.0E-6, 'limits': (-qtlimit5, qtlimit5)},
        'kq4.r8b2':    {'step': 1.0E-6, 'limits': (-qtlimit2, -qtlimit2*scmin)},
        'kq5.r8b2':    {'step': 1.0E-6, 'limits': ( qtlimit2*scmin, qtlimit2)},
        'kq6.r8b2':    {'step': 1.0E-6, 'limits': (-qtlimit2, -qtlimit2*scmin)},
        'kq7.r8b2':    {'step': 1.0E-6, 'limits': ( qtlimit3*scmin, qtlimit3)},
        'kq8.r8b2':    {'step': 1.0E-6, 'limits': (-qtlimit3, -qtlimit3*scmin)},
        'kq9.r8b2':    {'step': 1.0E-6, 'limits': ( qtlimit3*scmin, qtlimit3)},
        'kq10.r8b2':   {'step': 1.0E-6, 'limits': (-qtlimit3, -qtlimit3*scmin)},
        'kqtl11.r8b2': {'step': 1.0E-6, 'limits': (-qtlimit4, qtlimit4)},
        'kqt12.r8b2':  {'step': 1.0E-6, 'limits': (-qtlimit5, qtlimit5)},
        'kqt13.r8b2':  {'step': 1.0E-6, 'limits': (-qtlimit5, qtlimit5)},
        'kq5.l8b2':    {'step': 1.0E-6, 'limits': (-qtlimit2, -qtlimit2*scmin)},
        'kq4.l8b2':    {'step': 1.0E-6, 'limits': ( qtlimit2*scmin, qtlimit2)},
        'kq6.l8b2':    {'step': 1.0E-6, 'limits': ( qtlimit2*scmin, qtlimit2)},
        'kq7.l8b2':    {'step': 1.0E-6, 'limits': (-qtlimit3, -qtlimit3*scmin)},
        'kq8.l8b2':    {'step': 1.0E-6, 'limits': ( qtlimit3*scmin, qtlimit3)},
        'kq9.l8b2':    {'step': 1.0E-6, 'limits': (-qtlimit3, -qtlimit3*scmin)},
        'kq10.l8b2':   {'step': 1.0E-6, 'limits': ( qtlimit3*scmin, qtlimit3)},
        'kqtl11.l8b2': {'step': 1.0E-6, 'limits': (-qtlimit4, qtlimit4)},
        'kqt12.l8b2':  {'step': 1.0E-6, 'limits': (-qtlimit5, qtlimit5)},
        'kqt13.l8b2':  {'step': 1.0E-6, 'limits': (-qtlimit5, qtlimit5)},
    })


    mux_b1_target = 3.0985199176526272
    muy_b1_target = 2.7863674079923726

    mux_b2_target = 3.007814449420657
    muy_b2_target = 2.878419154545405

    collider.varval['kq6.l8b1'] *= 1.1
    collider.varval['kq6.r8b1'] *= 1.1

    tab_boundary_right = collider.lhcb1.twiss(
        start='ip8', end='ip1.l1',
        init=xt.TwissInit(element_name='ip1.l1', line=collider.lhcb1,
                                betx=0.15, bety=0.15))
    tab_boundary_left = collider.lhcb1.twiss(
        start='ip5', end='ip8',
        init=xt.TwissInit(element_name='ip5', line=collider.lhcb1,
                                betx=0.15, bety=0.15))

    opt = collider[f'lhcb1'].match(
        default_tol={None: 1e-7, 'betx': 1e-6, 'bety': 1e-6},
        solve=False,
        start=f's.ds.l8.b1', end=f'e.ds.r8.b1', init_at=xt.START,
        # Left boundary
        init=tab_boundary_left,
        targets=[
            xt.TargetSet(at='ip8', betx=1.5, bety=1.5, alfx=0, alfy=0, dx=0, dpx=0),
            xt.TargetSet(at=f'e.ds.r8.b1',
                         tars=('betx', 'bety', 'alfx', 'alfy', 'dx', 'dpx'),
                         value=tab_boundary_right, tag='stage2'),
            xt.TargetRelPhaseAdvance('mux', mux_b1_target),
            xt.TargetRelPhaseAdvance('muy', muy_b1_target),
        ],
        vary=[
            xt.VaryList([
                f'kq6.l8b1', f'kq7.l8b1',
                f'kq8.l8b1', f'kq9.l8b1', f'kq10.l8b1', f'kqtl11.l8b1',
                f'kqt12.l8b1', f'kqt13.l8b1']
            ),
            xt.Vary(f'kq4.l8b1', tag='stage1', step=1.1e-6, limits=(-0.007, -0.00022)),
            xt.Vary(f'kq5.l8b1', tag='stage1'),
            xt.VaryList([
                f'kq4.r8b1', f'kq5.r8b1', f'kq6.r8b1', f'kq7.r8b1',
                f'kq8.r8b1', f'kq9.r8b1', f'kq10.r8b1', f'kqtl11.r8b1',
                f'kqt12.r8b1', f'kqt13.r8b1'],
                tag='stage2')
        ]
    )

    # Initial knob values
    init_knob_vals = opt.get_knob_values()

    assert opt.vary[8].name == 'kq4.l8b1'
    assert opt.vary[8].tag == 'stage1'
    assert opt.vary[8].step == 1.1e-6
    assert opt.vary[8].limits[0] == -0.007
    assert opt.vary[8].limits[1] == -0.00022

    assert opt.vary[9].name == 'kq5.l8b1'
    assert opt.vary[9].tag == 'stage1'
    assert opt.vary[9].step == 1.0e-6
    assert opt.vary[9].limits[0] == qtlimit2*scmin
    assert opt.vary[9].limits[1] == qtlimit2

    # Assert that the two quad break the optics
    assert opt.log()['tol_met', 0] == 'nnnnnnnnnnnnnn'

    # Check that the unperturbed machine is a solution
    collider.varval['kq6.l8b1'] /= 1.1
    collider.varval['kq6.r8b1'] /= 1.1

    opt.clear_log()
    assert opt.log()['tol_met', 0] == 'yyyyyyyyyyyyyy'

    # Break again and clear log
    collider.varval['kq6.l8b1'] *= 1.1
    collider.varval['kq6.r8b1'] *= 1.1
    opt.clear_log()
    assert opt.log()['tol_met', 0] == 'nnnnnnnnnnnnnn'

    opt.disable(target=['stage1', 'stage2'])
    opt.tag()
    assert opt.log()['target_active', -1] == 'yyyyyynnnnnnyy'
    assert opt.log()['vary_active', -1] == 'yyyyyyyyyyyyyyyyyyyy'

    opt.disable(vary=['stage1', 'stage2'])
    opt.tag()
    assert opt.log()['target_active', -1] == 'yyyyyynnnnnnyy'
    assert opt.log()['vary_active', -1] == 'yyyyyyyynnnnnnnnnnnn'

    opt.step(12)
    assert opt.log()['penalty', -1] < 0.1
    assert opt.log()['target_active', -1] == 'yyyyyynnnnnnyy'
    assert opt.log()['vary_active', -1] == 'yyyyyyyynnnnnnnnnnnn'

    opt.enable(vary='stage1')
    opt.tag()
    assert opt.log()['target_active', -1] == 'yyyyyynnnnnnyy'
    assert opt.log()['vary_active', -1] == 'yyyyyyyyyynnnnnnnnnn'

    opt.enable(target='stage1')
    opt.tag()
    assert opt.log()['target_active', -1] == 'yyyyyynnnnnnyy'
    assert opt.log()['vary_active', -1] == 'yyyyyyyyyynnnnnnnnnn'

    opt.solve()
    assert opt.log()['penalty', -1] < 1e-7
    assert opt.log()['target_active', -1] == 'yyyyyynnnnnnyy'
    assert opt.log()['vary_active', -1] == 'yyyyyyyyyynnnnnnnnnn'

    opt.enable(target='stage2')
    opt.tag()
    assert opt.log()['target_active', -1] == 'yyyyyyyyyyyyyy'
    assert opt.log()['vary_active', -1] == 'yyyyyyyyyynnnnnnnnnn'
    assert opt.log()['tol_met', -1] != 'yyyyyyyyyyyyyy'

    opt.enable(vary='stage2')
    opt.tag()
    assert opt.log()['target_active', -1] == 'yyyyyyyyyyyyyy'
    assert opt.log()['vary_active', -1] == 'yyyyyyyyyyyyyyyyyyyy'

    opt.solve()
    opt.tag()
    assert opt.log()['target_active', -1] == 'yyyyyyyyyyyyyy'
    assert opt.log()['vary_active', -1] == 'yyyyyyyyyyyyyyyyyyyy'
    assert opt.log()['tol_met', -1] == 'yyyyyyyyyyyyyy'


    xo.assert_allclose(tw.lhcb1['betx', 'ip1'], 0.15, atol=1e-6, rtol=0)
    xo.assert_allclose(tw.lhcb1['bety', 'ip1'], 0.15, atol=1e-6, rtol=0)
    xo.assert_allclose(tw.lhcb2['betx', 'ip1'], 0.15, atol=1e-6, rtol=0)
    xo.assert_allclose(tw.lhcb2['bety', 'ip1'], 0.15, atol=1e-6, rtol=0)

    xo.assert_allclose(tw.lhcb1['betx', 'ip5'], 0.15, atol=1e-6, rtol=0)
    xo.assert_allclose(tw.lhcb1['bety', 'ip5'], 0.15, atol=1e-6, rtol=0)
    xo.assert_allclose(tw.lhcb2['betx', 'ip5'], 0.15, atol=1e-6, rtol=0)
    xo.assert_allclose(tw.lhcb2['bety', 'ip5'], 0.15, atol=1e-6, rtol=0)

    xo.assert_allclose(tw.lhcb1['betx', 'ip8'], 1.5, atol=1e-6, rtol=0)
    xo.assert_allclose(tw.lhcb1['bety', 'ip8'], 1.5, atol=1e-6, rtol=0)
    xo.assert_allclose(tw.lhcb2['betx', 'ip8'], 1.5, atol=1e-6, rtol=0)
    xo.assert_allclose(tw.lhcb2['bety', 'ip8'], 1.5, atol=1e-6, rtol=0)

    xo.assert_allclose(tw.lhcb1['betx', 'ip2'], 10., atol=1e-5, rtol=0)
    xo.assert_allclose(tw.lhcb1['bety', 'ip2'], 10., atol=1e-5, rtol=0)
    xo.assert_allclose(tw.lhcb2['betx', 'ip2'], 10., atol=1e-5, rtol=0)
    xo.assert_allclose(tw.lhcb2['bety', 'ip2'], 10., atol=1e-5, rtol=0)

    # Beam 2

    collider.varval['kq6.l8b2'] *= 1.1
    collider.varval['kq6.r8b2'] *= 1.1

    tab_boundary_right = collider.lhcb2.twiss(
        start='ip8', end='ip1.l1',
        init=xt.TwissInit(element_name='ip1.l1', line=collider.lhcb2,
                                betx=0.15, bety=0.15))
    tab_boundary_left = collider.lhcb2.twiss(
        start='ip5', end='ip8',
        init=xt.TwissInit(element_name='ip5', line=collider.lhcb2,
                                betx=0.15, bety=0.15))

    opt = collider[f'lhcb2'].match(
        default_tol={None: 1e-7, 'betx': 5e-6, 'bety': 5e-6},
        solve=False,
        start=f's.ds.l8.b2', end=f'e.ds.r8.b2', init_at=xt.START,
        # Left boundary
        init=tab_boundary_left,
        targets=[
            xt.TargetSet(at='ip8', betx=1.5, bety=1.5, alfx=0, alfy=0, dx=0, dpx=0),
            xt.TargetSet(at=xt.END,
                         tars=('betx', 'bety', 'alfx', 'alfy', 'dx', 'dpx'),
                         value=tab_boundary_right, tag='stage2'),
            xt.TargetRelPhaseAdvance('mux', mux_b2_target),
            xt.TargetRelPhaseAdvance('muy', muy_b2_target),
        ],
        vary=[
            xt.VaryList([
                f'kq6.l8b2', f'kq7.l8b2',
                f'kq8.l8b2', f'kq9.l8b2', f'kq10.l8b2', f'kqtl11.l8b2',
                f'kqt12.l8b2', f'kqt13.l8b2']
            ),
            xt.Vary(f'kq4.l8b2', tag='stage1', step=1.1e-6, limits=(0.00022, 0.007)),
            xt.Vary(f'kq5.l8b2', tag='stage1'),
            xt.VaryList([
                f'kq4.r8b2', f'kq5.r8b2', f'kq6.r8b2', f'kq7.r8b2',
                f'kq8.r8b2', f'kq9.r8b2', f'kq10.r8b2', f'kqtl11.r8b2',
                f'kqt12.r8b2', f'kqt13.r8b2'],
                tag='stage2')
        ]
    )

    assert opt.vary[8].name == 'kq4.l8b2'
    assert opt.vary[8].tag == 'stage1'
    assert opt.vary[8].step == 1.1e-6
    assert opt.vary[8].limits[0] == 0.00022
    assert opt.vary[8].limits[1] == 0.007

    assert opt.vary[9].name == 'kq5.l8b2'
    assert opt.vary[9].tag == 'stage1'
    assert opt.vary[9].step == 1.0e-6
    assert opt.vary[9].limits[0] == -qtlimit2
    assert opt.vary[9].limits[1] == -qtlimit2*scmin

    # Assert that the two quad break the optics
    assert opt.log()['tol_met', 0] == 'nnnnnnnnnnnnnn'

    # Check that the unperturbed machine is a solution
    collider.varval['kq6.l8b2'] /= 1.1
    collider.varval['kq6.r8b2'] /= 1.1

    opt.clear_log()
    assert opt.log()['tol_met', 0] == 'yyyyyyyyyyyyyy'

    # Break again and clear log
    collider.varval['kq6.l8b2'] *= 1.1
    collider.varval['kq6.r8b2'] *= 1.1
    opt.clear_log()
    assert opt.log()['tol_met', 0] == 'nnnnnnnnnnnnnn'

    opt.disable(target=['stage1', 'stage2'])
    opt.tag()
    assert opt.log()['target_active', -1] == 'yyyyyynnnnnnyy'
    assert opt.log()['vary_active', -1] == 'yyyyyyyyyyyyyyyyyyyy'

    opt.disable(vary=['stage1', 'stage2'])
    opt.tag()
    assert opt.log()['target_active', -1] == 'yyyyyynnnnnnyy'
    assert opt.log()['vary_active', -1] == 'yyyyyyyynnnnnnnnnnnn'

    # Tag present state
    knob_values_before_tag0 = opt.get_knob_values()
    i_iter_tag0 = opt.log().iteration[-1]

    opt.step(10)
    assert opt.log()['penalty', -1] < 0.1
    assert opt.log()['target_active', -1] == 'yyyyyynnnnnnyy'
    assert opt.log()['vary_active', -1] == 'yyyyyyyynnnnnnnnnnnn'

    # Tag present state
    knob_values_before_tag1 = opt.get_knob_values()
    opt.tag(tag='mytag1')

    ##### Check of reloading features #####

    # Check that knobs have changed
    assert np.any([knob_values_before_tag0[k] != knob_values_before_tag1[k]
                     for k in knob_values_before_tag0])

    # Reload with iteration number
    opt.reload(iteration=i_iter_tag0)
    knobs_now = opt.get_knob_values()
    assert np.all([knobs_now[k] == knob_values_before_tag0[k]
                        for k in knob_values_before_tag0])

    # Reload with tag
    opt.reload(tag='mytag1')
    knobs_now = opt.get_knob_values()
    assert np.all([knobs_now[k] == knob_values_before_tag1[k]
                        for k in knob_values_before_tag1])

    ##### Done checking reloading features #####

    opt.enable(vary='stage1')
    opt.tag()
    assert opt.log()['target_active', -1] == 'yyyyyynnnnnnyy'
    assert opt.log()['vary_active', -1] == 'yyyyyyyyyynnnnnnnnnn'

    opt.enable(target='stage1')
    opt.tag()
    assert opt.log()['target_active', -1] == 'yyyyyynnnnnnyy'
    assert opt.log()['vary_active', -1] == 'yyyyyyyyyynnnnnnnnnn'

    opt.solve()
    assert opt.log()['penalty', -1] < 1e-5
    assert opt.log()['target_active', -1] == 'yyyyyynnnnnnyy'
    assert opt.log()['vary_active', -1] == 'yyyyyyyyyynnnnnnnnnn'

    opt.enable(target='stage2')
    opt.tag()
    assert opt.log()['target_active', -1] == 'yyyyyyyyyyyyyy'
    assert opt.log()['vary_active', -1] == 'yyyyyyyyyynnnnnnnnnn'
    assert opt.log()['tol_met', -1] != 'yyyyyyyyyyyyyy'

    opt.enable(vary='stage2')
    opt.tag()
    assert opt.log()['target_active', -1] == 'yyyyyyyyyyyyyy'
    assert opt.log()['vary_active', -1] == 'yyyyyyyyyyyyyyyyyyyy'

    opt.solve()
    opt.tag()
    assert opt.log()['target_active', -1] == 'yyyyyyyyyyyyyy'
    assert opt.log()['vary_active', -1] == 'yyyyyyyyyyyyyyyyyyyy'
    assert opt.log()['tol_met', -1] == 'yyyyyyyyyyyyyy'

    xo.assert_allclose(tw.lhcb1['betx', 'ip1'], 0.15, atol=1e-6, rtol=0)
    xo.assert_allclose(tw.lhcb1['bety', 'ip1'], 0.15, atol=1e-6, rtol=0)
    xo.assert_allclose(tw.lhcb2['betx', 'ip1'], 0.15, atol=1e-6, rtol=0)
    xo.assert_allclose(tw.lhcb2['bety', 'ip1'], 0.15, atol=1e-6, rtol=0)

    xo.assert_allclose(tw.lhcb1['betx', 'ip5'], 0.15, atol=1e-6, rtol=0)
    xo.assert_allclose(tw.lhcb1['bety', 'ip5'], 0.15, atol=1e-6, rtol=0)
    xo.assert_allclose(tw.lhcb2['betx', 'ip5'], 0.15, atol=1e-6, rtol=0)
    xo.assert_allclose(tw.lhcb2['bety', 'ip5'], 0.15, atol=1e-6, rtol=0)

    xo.assert_allclose(tw.lhcb1['betx', 'ip8'], 1.5, atol=1e-6, rtol=0)
    xo.assert_allclose(tw.lhcb1['bety', 'ip8'], 1.5, atol=1e-6, rtol=0)
    xo.assert_allclose(tw.lhcb2['betx', 'ip8'], 1.5, atol=1e-6, rtol=0)
    xo.assert_allclose(tw.lhcb2['bety', 'ip8'], 1.5, atol=1e-6, rtol=0)

    xo.assert_allclose(tw.lhcb1['betx', 'ip2'], 10., atol=1e-5, rtol=0)
    xo.assert_allclose(tw.lhcb1['bety', 'ip2'], 10., atol=1e-5, rtol=0)
    xo.assert_allclose(tw.lhcb2['betx', 'ip2'], 10., atol=1e-5, rtol=0)
    xo.assert_allclose(tw.lhcb2['bety', 'ip2'], 10., atol=1e-5, rtol=0)

