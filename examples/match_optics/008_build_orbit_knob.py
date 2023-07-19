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

# Match IP offset knob
offset_match = 0.5e-3
opt = collider.match_knob(
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
opt.solve()
opt.generate_knob()

collider.vars['on_o2v'] = 0.3
tw = collider.twiss()
collider.vars['on_o2v'] = 0

assert np.isclose(tw.lhcb1['y', 'ip2'], 0.3e-3, atol=1e-10, rtol=0)
assert np.isclose(tw.lhcb2['y', 'ip2'], 0.3e-3, atol=1e-10, rtol=0)
assert np.isclose(tw.lhcb1['py', 'ip2'], 0., atol=1e-10, rtol=0)
assert np.isclose(tw.lhcb2['py', 'ip2'], 0., atol=1e-10, rtol=0)

# Match crossing angle knob
angle_match = 170e-6
opt = collider.match_knob(
    run=False,
    knob_name='on_x2v',
    knob_value_start=0,
    knob_value_end=(angle_match * 1e6),
    ele_start=['s.ds.l2.b1', 's.ds.l2.b2'],
    ele_stop=['e.ds.r2.b1', 'e.ds.r2.b2'],
    twiss_init=[xt.TwissInit(betx=1, bety=1, element_name='s.ds.l2.b1', line=collider.lhcb1),
                xt.TwissInit(betx=1, bety=1, element_name='s.ds.l2.b2', line=collider.lhcb2)],
    targets=[
        xt.TargetList(['y', 'py'], at='e.ds.r2.b1', line='lhcb1', value=0),
        xt.TargetList(['y', 'py'], at='e.ds.r2.b2', line='lhcb2', value=0),
        xt.Target('y', 0, at='ip2', line='lhcb1'),
        xt.Target('y', 0, at='ip2', line='lhcb2'),
        xt.Target('py', angle_match, at='ip2', line='lhcb1'),
        xt.Target('py', -angle_match, at='ip2', line='lhcb2'),
    ],
    vary=[
        xt.VaryList([
            'acbyvs4.l2b1', 'acbyvs4.r2b2', 'acbyvs4.l2b2', 'acbyvs4.r2b1',
            'acbyvs5.l2b2', 'acbyvs5.l2b1', 'acbcvs5.r2b1', 'acbcvs5.r2b2']),
          xt.VaryList([
            'acbxv1.l2', 'acbxv2.l2', 'acbxv3.l2',
            'acbxv1.r2', 'acbxv2.r2', 'acbxv3.r2'], tag='mcbx')]
)

# Set mcmbx by hand
testkqx2=abs(collider.varval['kqx.l2'])*7000./0.3
if testkqx2> 210.:
    acbx_xing_ir2 = 1.0e-6   # Value for 170 urad crossing
else:
    acbx_xing_ir2 = 11.0e-6  # Value for 170 urad crossing

collider.vars['acbxv1.l2_from_on_x2v'] = acbx_xing_ir2
collider.vars['acbxv2.l2_from_on_x2v'] = acbx_xing_ir2
collider.vars['acbxv3.l2_from_on_x2v'] = acbx_xing_ir2
collider.vars['acbxv1.r2_from_on_x2v'] = -acbx_xing_ir2
collider.vars['acbxv2.r2_from_on_x2v'] = -acbx_xing_ir2
collider.vars['acbxv3.r2_from_on_x2v'] = -acbx_xing_ir2

# match other knobs with fixed mcbx
opt.disable_vary(tag='mcbx')
opt.solve()
opt.generate_knob()

collider.vars['on_x2v'] = 100
tw = collider.twiss()
collider.vars['on_x2v'] = 0

assert np.isclose(tw.lhcb1['y', 'ip2'], 0, atol=1e-10, rtol=0)
assert np.isclose(tw.lhcb2['y', 'ip2'], 0, atol=1e-10, rtol=0)
assert np.isclose(tw.lhcb1['py', 'ip2'], 100e-6, atol=1e-10, rtol=0)
assert np.isclose(tw.lhcb2['py', 'ip2'], -100e-6, atol=1e-10, rtol=0)

# Match horizontal xing angle in ip8

angle_match = 300e-6
opt = collider.match_knob(
    run=False,
    knob_name='on_x8h',
    knob_value_start=0,
    knob_value_end=(angle_match * 1e6),
    ele_start=['s.ds.l8.b1', 's.ds.l8.b2'],
    ele_stop=['e.ds.r8.b1', 'e.ds.r8.b2'],
    twiss_init=[xt.TwissInit(betx=1, bety=1, element_name='s.ds.l8.b1', line=collider.lhcb1),
                xt.TwissInit(betx=1, bety=1, element_name='s.ds.l8.b2', line=collider.lhcb2)],
    targets=[
        xt.TargetList(['x', 'px'], at='e.ds.r8.b1', line='lhcb1', value=0),
        xt.TargetList(['x', 'px'], at='e.ds.r8.b2', line='lhcb2', value=0),
        xt.Target('x', 0, at='ip8', line='lhcb1'),
        xt.Target('x', 0, at='ip8', line='lhcb2'),
        xt.Target('px', angle_match, at='ip8', line='lhcb1'),
        xt.Target('px', -angle_match, at='ip8', line='lhcb2'),
    ],
    vary=[
        xt.VaryList([
            'acbyhs4.l8b1', 'acbyhs4.r8b2', 'acbyhs4.l8b2', 'acbyhs4.r8b1',
            'acbchs5.l8b2', 'acbchs5.l8b1', 'acbyhs5.r8b1', 'acbyhs5.r8b2']),
        xt.VaryList([
            'acbxh1.l8', 'acbxh2.l8', 'acbxh3.l8',
            'acbxh1.r8', 'acbxh2.r8', 'acbxh3.r8'], tag='mcbx')]
    )

# Set mcmbx by hand
testkqx8=abs(collider.varval['kqx.l8'])*7000./0.3
if testkqx8> 210.:
    acbx_xing_ir8 = 1.0e-6   # Value for 170 urad crossing
else:
    acbx_xing_ir8 = 11.0e-6  # Value for 170 urad crossing

collider.vars['acbxh1.l8_from_on_x8h'] = acbx_xing_ir8 * angle_match / 170e-6
collider.vars['acbxh2.l8_from_on_x8h'] = acbx_xing_ir8 * angle_match / 170e-6
collider.vars['acbxh3.l8_from_on_x8h'] = acbx_xing_ir8 * angle_match / 170e-6
collider.vars['acbxh1.r8_from_on_x8h'] = -acbx_xing_ir8 * angle_match / 170e-6
collider.vars['acbxh2.r8_from_on_x8h'] = -acbx_xing_ir8 * angle_match / 170e-6
collider.vars['acbxh3.r8_from_on_x8h'] = -acbx_xing_ir8 * angle_match / 170e-6

# Firs round of optimization without changing mcbx
opt.disable_vary(tag='mcbx')

opt.step(10)

# Link all mcbx stengths to the first one
collider.vars['acbxh2.l8_from_on_x8h'] =  collider.vars['acbxh1.l8_from_on_x8h']
collider.vars['acbxh3.l8_from_on_x8h'] =  collider.vars['acbxh1.l8_from_on_x8h']
collider.vars['acbxh2.r8_from_on_x8h'] = -collider.vars['acbxh1.l8_from_on_x8h']
collider.vars['acbxh3.r8_from_on_x8h'] = -collider.vars['acbxh1.l8_from_on_x8h']
collider.vars['acbxh1.r8_from_on_x8h'] = -collider.vars['acbxh1.l8_from_on_x8h']

# Enable first mcbx knob
assert opt.vary[8].name == 'acbxh1.l8_from_on_x8h'
opt.vary[8].active = True


opt.solve()