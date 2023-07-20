import numpy as np

import xtrack as xt
import lhc_match as lm


collider = xt.Multiline.from_json(
    '../../test_data/hllhc15_thick/hllhc15_collider_thick.json')
collider.build_trackers()

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

assert np.isclose(collider.vars.vary_default['acbxh1.l8']['step'], 1.0e-15, atol=1e-17, rtol=0)
assert np.isclose(collider.vars.vary_default['acbxh1.l8']['limits'][0], -limitmcbx, atol=1e-17, rtol=0)
assert np.isclose(collider.vars.vary_default['acbxh1.l8']['limits'][1], limitmcbx, atol=1e-17, rtol=0)
assert np.isclose(collider.vars.vary_default['acbyhs5.r8b2']['step'], 1.0e-15, atol=1e-17, rtol=0)
assert np.isclose(collider.vars.vary_default['acbyhs5.r8b2']['limits'][0], -limitmcbc, atol=1e-17, rtol=0)
assert np.isclose(collider.vars.vary_default['acbyhs5.r8b2']['limits'][1], limitmcbc, atol=1e-17, rtol=0)

# Check that they are preserved by to_dict/from_dict
collider = xt.Multiline.from_dict(collider.to_dict())
collider.build_trackers()

assert np.isclose(collider.vars.vary_default['acbxh1.l8']['step'], 1.0e-15, atol=1e-17, rtol=0)
assert np.isclose(collider.vars.vary_default['acbxh1.l8']['limits'][0], -limitmcbx, atol=1e-17, rtol=0)
assert np.isclose(collider.vars.vary_default['acbxh1.l8']['limits'][1], limitmcbx, atol=1e-17, rtol=0)
assert np.isclose(collider.vars.vary_default['acbyhs5.r8b2']['step'], 1.0e-15, atol=1e-17, rtol=0)
assert np.isclose(collider.vars.vary_default['acbyhs5.r8b2']['limits'][0], -limitmcbc, atol=1e-17, rtol=0)
assert np.isclose(collider.vars.vary_default['acbyhs5.r8b2']['limits'][1], limitmcbc, atol=1e-17, rtol=0)

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
    ele_start=['s.ds.l8.b1', 's.ds.l8.b2'],
    ele_stop=['e.ds.r8.b1', 'e.ds.r8.b2'],
    twiss_init=[xt.TwissInit(betx=1, bety=1, element_name='s.ds.l8.b1', line=collider.lhcb1),
                xt.TwissInit(betx=1, bety=1, element_name='s.ds.l8.b2', line=collider.lhcb2)],
    targets=[
        xt.TargetList(['x', 'px'], at='e.ds.r8.b1', line='lhcb1', value=0),
        xt.TargetList(['x', 'px'], at='e.ds.r8.b2', line='lhcb2', value=0),
        xt.Target('x', 0, at='ip8', line='lhcb1'),
        xt.Target('x', 0, at='ip8', line='lhcb2'),
        xt.Target('px', angle_match, at='ip8', line='lhcb1', tol=0.8e-10),
        xt.Target('px', -angle_match, at='ip8', line='lhcb2'),
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
assert np.isclose(ll['penalty', 0], 0.0424264, atol=1e-6, rtol=0)
assert ll['tol_met', 0] == 'yyyyyynn'
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

assert np.isclose(opt.vary[0].step, 2e-15, atol=1e-17, rtol=0)
assert np.isclose(opt.vary[0].limits[0], -9e-5, atol=1e-10, rtol=0)
assert np.isclose(opt.vary[0].limits[1], 9e-5, atol=1e-10, rtol=0)
assert np.isclose(opt.vary[2].step, 1e-15, atol=1e-17, rtol=0)
assert np.isclose(opt.vary[2].limits[0], -limitmcby, atol=1e-10, rtol=0)
assert np.isclose(opt.vary[2].limits[1], limitmcby, atol=1e-10, rtol=0)

assert np.isclose(opt.targets[5].tol, 1.1e-10, atol=1e-14, rtol=0)
assert np.isclose(opt.targets[6].tol, 0.8e-10, atol=1e-14, rtol=0)
assert np.isclose(opt.targets[7].tol, 0.9e-10, atol=1e-14, rtol=0)

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
opt.disable_vary(tag='mcbx')
vtags = [vv.tag for vv in opt.vary]
assert np.all(np.array(vtags) == np.array(
    ['', '', '', '', '', '', '', '', 'mcbx', 'mcbx', 'mcbx', 'mcbx', 'mcbx', 'mcbx']))
vactive = [vv.active for vv in opt.vary]
assert np.all(np.array(vactive) == np.array(
    [True, True, True, True, True, True, True, True, False, False, False, False, False, False]))

opt.step(10) # perform 10 steps without checking for convergence

ll = opt.log()
assert len(ll) == 11
assert ll['vary_active', 0] == 'yyyyyyyyyyyyyy'
assert ll['vary_active', 1] == 'yyyyyyyynnnnnn'
assert ll['vary_active', 10] == 'yyyyyyyynnnnnn'

# Check solution not found
assert ll['tol_met', 10] != 'yyyyyyyy'

# Check that mcbxs did not move
assert np.allclose(ll['vary_8', 1:], init_mcbx_plus, atol=1e-12, rtol=0)
assert np.allclose(ll['vary_9', 1:], init_mcbx_plus, atol=1e-12, rtol=0)
assert np.allclose(ll['vary_10', 1:], init_mcbx_plus, atol=1e-12, rtol=0)
assert np.allclose(ll['vary_11', 1:], -init_mcbx_plus, atol=1e-12, rtol=0)
assert np.allclose(ll['vary_12', 1:], -init_mcbx_plus, atol=1e-12, rtol=0)
assert np.allclose(ll['vary_13', 1:], -init_mcbx_plus, atol=1e-12, rtol=0)


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
assert np.all(ll['vary_active', 11:] == 'yyyyyyyyynnnnn')

# Check solution found
assert np.all(ll['tol_met', -1] == 'yyyyyyyy')

# Check imposed relationship among varys
assert np.isclose(ll['vary_8', 11], ll['vary_9', 11], atol=1e-12, rtol=0)
assert np.isclose(ll['vary_8', 11], ll['vary_10', 11], atol=1e-12, rtol=0)
assert np.isclose(ll['vary_8', 11], -ll['vary_11', 11], atol=1e-12, rtol=0)
assert np.isclose(ll['vary_8', 11], -ll['vary_12', 11], atol=1e-12, rtol=0)
assert np.isclose(ll['vary_8', 11], -ll['vary_13', 11], atol=1e-12, rtol=0)

opt.generate_knob()

collider.vars['on_x8h'] = 100
tw = collider.twiss()
collider.vars['on_x8h'] = 0

assert np.isclose(tw.lhcb1['x', 'ip8'], 0, atol=1e-10, rtol=0)
assert np.isclose(tw.lhcb2['x', 'ip8'], 0, atol=1e-10, rtol=0)
assert np.isclose(tw.lhcb1['px', 'ip8'], 100e-6, atol=1e-10, rtol=0)
assert np.isclose(tw.lhcb2['px', 'ip8'], -100e-6, atol=1e-10, rtol=0)

# Match horizontal separation in ip8
sep_match = 2e-3
opt = collider.match_knob(
    run=False,
    knob_name='on_sep8h',
    knob_value_start=0,
    knob_value_end=(sep_match * 1e3),
    ele_start=['s.ds.l8.b1', 's.ds.l8.b2'],
    ele_stop=['e.ds.r8.b1', 'e.ds.r8.b2'],
    twiss_init=[xt.TwissInit(betx=1, bety=1, element_name='s.ds.l8.b1', line=collider.lhcb1),
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
opt.disable_vary(tag='mcbx')
opt.step(10) # perform 10 steps without checking for convergence

# Enable first mcbx knob (which controls the others)
assert opt.vary[8].name == 'acbxh1.l8_from_on_sep8h'
opt.vary[8].active = True

opt.solve()
opt.generate_knob()

collider.vars['on_sep8h'] = 1.5
tw = collider.twiss()
collider.vars['on_sep8h'] = 0

assert np.isclose(tw.lhcb1['x', 'ip8'], 1.5e-3, atol=1e-10, rtol=0)
assert np.isclose(tw.lhcb2['x', 'ip8'], -1.5e-3, atol=1e-10, rtol=0)
assert np.isclose(tw.lhcb1['px', 'ip8'], 0, atol=1e-10, rtol=0)
assert np.isclose(tw.lhcb2['px', 'ip8'], 0, atol=1e-10, rtol=0)

# Check that on_x8h still works
collider.vars['on_x8h'] = 100
tw = collider.twiss()
collider.vars['on_x8h'] = 0

assert np.isclose(tw.lhcb1['x', 'ip8'], 0, atol=1e-10, rtol=0)
assert np.isclose(tw.lhcb2['x', 'ip8'], 0, atol=1e-10, rtol=0)
assert np.isclose(tw.lhcb1['px', 'ip8'], 100e-6, atol=1e-10, rtol=0)
assert np.isclose(tw.lhcb2['px', 'ip8'], -100e-6, atol=1e-10, rtol=0)

# Both knobs together
collider.vars['on_x8h'] = 120
collider.vars['on_sep8h'] = 1.7
tw = collider.twiss()
collider.vars['on_x8h'] = 0
collider.vars['on_sep8h'] = 0

assert np.isclose(tw.lhcb1['x', 'ip8'], 1.7e-3, atol=1e-10, rtol=0)
assert np.isclose(tw.lhcb2['x', 'ip8'], -1.7e-3, atol=1e-10, rtol=0)
assert np.isclose(tw.lhcb1['px', 'ip8'], 120e-6, atol=1e-10, rtol=0)
assert np.isclose(tw.lhcb2['px', 'ip8'], -120e-6, atol=1e-10, rtol=0)
