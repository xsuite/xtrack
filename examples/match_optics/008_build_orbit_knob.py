import numpy as np

import xtrack as xt
import lhc_match as lm

default_tol = {None: 1e-8, 'betx': 1e-6, 'bety': 1e-6} # to have no rematching w.r.t. madx

collider = xt.Multiline.from_json('hllhc.json')
collider.build_trackers()
collider.vars.load_madx_optics_file(
    "../../test_data/hllhc15_thick/opt_round_150_1500.madx")

tw0 = collider.twiss()

collider0 = collider.copy()
collider0.build_trackers()

all_knobs_ip2ip8 = ['acbxh3.r2', 'acbchs5.r2b1', 'pxip2b1', 'acbxh2.l8',
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

# kill all existing knobs
for kk in all_knobs_ip2ip8:
    collider.vars[kk] = 0

# We start by matching a bump, no knob

offset_match = 0.5e-3
knob_opt = collider.match_knob(
    run=False,
    knob_name='on_o2v',
    knob_value_start=0,
    knob_value_end=(offset_match * 1e3),
    ele_start=['s.ds.l2.b1', 's.ds.l2.b2'],
    ele_stop=['e.ds.r2.b1', 'e.ds.r2.b2'],
    twiss_init=[xt.TwissInit(betx=1, bety=1, element_name='s.ds.l2.b1', line=collider.lhcb1),
                xt.TwissInit(betx=1, bety=1, element_name='s.ds.l2.b2', line=collider.lhcb2)],
    targets=[
        xt.TargetList(['y', 'py'], at='e.ds.r2.b1', line='lhcb1', value=0),
        xt.TargetList(['y', 'py'], at='e.ds.r2.b2', line='lhcb2', value=0),
        xt.Target('y', offset_match, at='ip2', line='lhcb1'),
        xt.Target('y', offset_match, at='ip2', line='lhcb2'),
        xt.Target('py', 0., at='ip2', line='lhcb1'),
        xt.Target('py', 0., at='ip2', line='lhcb2'),
    ],
    vary=xt.VaryList([
        'acbyvs4.l2b1', 'acbyvs4.r2b2', 'acbyvs4.l2b2', 'acbyvs4.r2b1',
        'acbyvs5.l2b2', 'acbyvs5.l2b1', 'acbcvs5.r2b1', 'acbcvs5.r2b2']),
)
knob_opt.solve()
knob_opt.generate_knob()

collider.vars['on_o2v'] = 0.3
tw = collider.twiss()

assert np.isclose(tw.lhcb1['y', 'ip2'], 0.3e-3, atol=1e-10, rtol=0)
assert np.isclose(tw.lhcb2['y', 'ip2'], 0.3e-3, atol=1e-10, rtol=0)
assert np.isclose(tw.lhcb1['py', 'ip2'], 0., atol=1e-10, rtol=0)
assert np.isclose(tw.lhcb2['py', 'ip2'], 0., atol=1e-10, rtol=0)

