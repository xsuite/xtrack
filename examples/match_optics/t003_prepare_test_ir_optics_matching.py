import numpy as np
import xtrack as xt

collider = xt.Multiline.from_json(
    '../../test_data/hllhc15_thick/hllhc15_collider_thick.json')
collider.build_trackers()

tw = collider.twiss()
assert np.isclose(tw.lhcb1['betx', 'ip1'], 0.15, atol=1e-6, rtol=0)
assert np.isclose(tw.lhcb1['bety', 'ip1'], 0.15, atol=1e-6, rtol=0)
assert np.isclose(tw.lhcb2['betx', 'ip1'], 0.15, atol=1e-6, rtol=0)
assert np.isclose(tw.lhcb2['bety', 'ip1'], 0.15, atol=1e-6, rtol=0)

assert np.isclose(tw.lhcb1['betx', 'ip5'], 0.15, atol=1e-6, rtol=0)
assert np.isclose(tw.lhcb1['bety', 'ip5'], 0.15, atol=1e-6, rtol=0)
assert np.isclose(tw.lhcb2['betx', 'ip5'], 0.15, atol=1e-6, rtol=0)
assert np.isclose(tw.lhcb2['bety', 'ip5'], 0.15, atol=1e-6, rtol=0)

assert np.isclose(tw.lhcb1['betx', 'ip8'], 1.5, atol=1e-6, rtol=0)
assert np.isclose(tw.lhcb1['bety', 'ip8'], 1.5, atol=1e-6, rtol=0)
assert np.isclose(tw.lhcb2['betx', 'ip8'], 1.5, atol=1e-6, rtol=0)
assert np.isclose(tw.lhcb2['bety', 'ip8'], 1.5, atol=1e-6, rtol=0)

assert np.isclose(tw.lhcb1['betx', 'ip2'], 10., atol=1e-5, rtol=0)
assert np.isclose(tw.lhcb1['bety', 'ip2'], 10., atol=1e-5, rtol=0)
assert np.isclose(tw.lhcb2['betx', 'ip2'], 10., atol=1e-5, rtol=0)
assert np.isclose(tw.lhcb2['bety', 'ip2'], 10., atol=1e-5, rtol=0)






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
    ele_start='ip8', ele_stop='ip1.l1',
    twiss_init=xt.TwissInit(element_name='ip1.l1', line=collider.lhcb1,
                            betx=0.15, bety=0.15))
tab_boundary_left = collider.lhcb1.twiss(
    ele_start='ip5', ele_stop='ip8',
    twiss_init=xt.TwissInit(element_name='ip5', line=collider.lhcb1,
                            betx=0.15, bety=0.15))

opt = collider[f'lhcb1'].match(
    default_tol={None: 1e-7, 'betx': 1e-6, 'bety': 1e-6},
    solve=False,
    ele_start=f's.ds.l8.b1', ele_stop=f'e.ds.r8.b1',
    # Left boundary
    twiss_init='preserve_start', table_for_twiss_init=tab_boundary_left,
    targets=[
        xt.Target('alfx', 0, at='ip8'),
        xt.Target('alfy', 0, at='ip8'),
        xt.Target('betx', 1.5, at='ip8'),
        xt.Target('bety', 1.5, at='ip8'),
        xt.Target('dx', 0, at='ip8'),
        xt.Target('dpx', 0, at='ip8'),
        xt.TargetList(('betx', 'bety', 'alfx', 'alfy', 'dx', 'dpx'),
                value=tab_boundary_right, at=f'e.ds.r8.b1',
                tag='stage2'),
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

opt.disable_targets(tag=['stage1', 'stage2'])
opt._add_point_to_log()
assert opt.log()['target_active', -1] == 'yyyyyynnnnnnyy'
assert opt.log()['vary_active', -1] == 'yyyyyyyyyyyyyyyyyyyy'

opt.disable_vary(tag=['stage1', 'stage2'])
opt._add_point_to_log()
assert opt.log()['target_active', -1] == 'yyyyyynnnnnnyy'
assert opt.log()['vary_active', -1] == 'yyyyyyyynnnnnnnnnnnn'

opt.step(10)
assert opt.log()['penalty', -1] < 0.1
assert opt.log()['target_active', -1] == 'yyyyyynnnnnnyy'
assert opt.log()['vary_active', -1] == 'yyyyyyyynnnnnnnnnnnn'

opt.enable_vary(tag='stage1')
opt._add_point_to_log()
assert opt.log()['target_active', -1] == 'yyyyyynnnnnnyy'
assert opt.log()['vary_active', -1] == 'yyyyyyyyyynnnnnnnnnn'

opt.enable_targets(tag='stage1')
opt._add_point_to_log()
assert opt.log()['target_active', -1] == 'yyyyyynnnnnnyy'
assert opt.log()['vary_active', -1] == 'yyyyyyyyyynnnnnnnnnn'

opt.solve()
assert opt.log()['penalty', -1] < 1e-7
assert opt.log()['target_active', -1] == 'yyyyyynnnnnnyy'
assert opt.log()['vary_active', -1] == 'yyyyyyyyyynnnnnnnnnn'

opt.enable_targets(tag='stage2')
opt._add_point_to_log()
assert opt.log()['target_active', -1] == 'yyyyyyyyyyyyyy'
assert opt.log()['vary_active', -1] == 'yyyyyyyyyynnnnnnnnnn'
assert opt.log()['tol_met', -1] != 'yyyyyyyyyyyyyy'

opt.enable_vary(tag='stage2')
opt._add_point_to_log()
assert opt.log()['target_active', -1] == 'yyyyyyyyyyyyyy'
assert opt.log()['vary_active', -1] == 'yyyyyyyyyyyyyyyyyyyy'

opt.solve()
opt._add_point_to_log()
assert opt.log()['target_active', -1] == 'yyyyyyyyyyyyyy'
assert opt.log()['vary_active', -1] == 'yyyyyyyyyyyyyyyyyyyy'
assert opt.log()['tol_met', -1] == 'yyyyyyyyyyyyyy'


assert np.isclose(tw.lhcb1['betx', 'ip1'], 0.15, atol=1e-6, rtol=0)
assert np.isclose(tw.lhcb1['bety', 'ip1'], 0.15, atol=1e-6, rtol=0)
assert np.isclose(tw.lhcb2['betx', 'ip1'], 0.15, atol=1e-6, rtol=0)
assert np.isclose(tw.lhcb2['bety', 'ip1'], 0.15, atol=1e-6, rtol=0)

assert np.isclose(tw.lhcb1['betx', 'ip5'], 0.15, atol=1e-6, rtol=0)
assert np.isclose(tw.lhcb1['bety', 'ip5'], 0.15, atol=1e-6, rtol=0)
assert np.isclose(tw.lhcb2['betx', 'ip5'], 0.15, atol=1e-6, rtol=0)
assert np.isclose(tw.lhcb2['bety', 'ip5'], 0.15, atol=1e-6, rtol=0)

assert np.isclose(tw.lhcb1['betx', 'ip8'], 1.5, atol=1e-6, rtol=0)
assert np.isclose(tw.lhcb1['bety', 'ip8'], 1.5, atol=1e-6, rtol=0)
assert np.isclose(tw.lhcb2['betx', 'ip8'], 1.5, atol=1e-6, rtol=0)
assert np.isclose(tw.lhcb2['bety', 'ip8'], 1.5, atol=1e-6, rtol=0)

assert np.isclose(tw.lhcb1['betx', 'ip2'], 10., atol=1e-5, rtol=0)
assert np.isclose(tw.lhcb1['bety', 'ip2'], 10., atol=1e-5, rtol=0)
assert np.isclose(tw.lhcb2['betx', 'ip2'], 10., atol=1e-5, rtol=0)
assert np.isclose(tw.lhcb2['bety', 'ip2'], 10., atol=1e-5, rtol=0)

# Beam 2

collider.varval['kq6.l8b2'] *= 1.1
collider.varval['kq6.r8b2'] *= 1.1

tab_boundary_right = collider.lhcb2.twiss(
    ele_start='ip8', ele_stop='ip1.l1',
    twiss_init=xt.TwissInit(element_name='ip1.l1', line=collider.lhcb2,
                            betx=0.15, bety=0.15))
tab_boundary_left = collider.lhcb2.twiss(
    ele_start='ip5', ele_stop='ip8',
    twiss_init=xt.TwissInit(element_name='ip5', line=collider.lhcb2,
                            betx=0.15, bety=0.15))

opt = collider[f'lhcb2'].match(
    default_tol={None: 1e-7, 'betx': 1e-6, 'bety': 1e-6},
    solve=False,
    ele_start=f's.ds.l8.b2', ele_stop=f'e.ds.r8.b2',
    # Left boundary
    twiss_init='preserve_start', table_for_twiss_init=tab_boundary_left,
    targets=[
        xt.Target('alfx', 0, at='ip8'),
        xt.Target('alfy', 0, at='ip8'),
        xt.Target('betx', 1.5, at='ip8'),
        xt.Target('bety', 1.5, at='ip8'),
        xt.Target('dx', 0, at='ip8'),
        xt.Target('dpx', 0, at='ip8'),
        xt.TargetList(('betx', 'bety', 'alfx', 'alfy', 'dx', 'dpx'),
                value=tab_boundary_right, at=f'e.ds.r8.b2',
                tag='stage2'),
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

opt.disable_targets(tag=['stage1', 'stage2'])
opt._add_point_to_log()
assert opt.log()['target_active', -1] == 'yyyyyynnnnnnyy'
assert opt.log()['vary_active', -1] == 'yyyyyyyyyyyyyyyyyyyy'

opt.disable_vary(tag=['stage1', 'stage2'])
opt._add_point_to_log()
assert opt.log()['target_active', -1] == 'yyyyyynnnnnnyy'
assert opt.log()['vary_active', -1] == 'yyyyyyyynnnnnnnnnnnn'

opt.step(10)
assert opt.log()['penalty', -1] < 0.1
assert opt.log()['target_active', -1] == 'yyyyyynnnnnnyy'
assert opt.log()['vary_active', -1] == 'yyyyyyyynnnnnnnnnnnn'

opt.enable_vary(tag='stage1')
opt._add_point_to_log()
assert opt.log()['target_active', -1] == 'yyyyyynnnnnnyy'
assert opt.log()['vary_active', -1] == 'yyyyyyyyyynnnnnnnnnn'

opt.enable_targets(tag='stage1')
opt._add_point_to_log()
assert opt.log()['target_active', -1] == 'yyyyyynnnnnnyy'
assert opt.log()['vary_active', -1] == 'yyyyyyyyyynnnnnnnnnn'

opt.solve()
assert opt.log()['penalty', -1] < 1e-5
assert opt.log()['target_active', -1] == 'yyyyyynnnnnnyy'
assert opt.log()['vary_active', -1] == 'yyyyyyyyyynnnnnnnnnn'

opt.enable_targets(tag='stage2')
opt._add_point_to_log()
assert opt.log()['target_active', -1] == 'yyyyyyyyyyyyyy'
assert opt.log()['vary_active', -1] == 'yyyyyyyyyynnnnnnnnnn'
assert opt.log()['tol_met', -1] != 'yyyyyyyyyyyyyy'

opt.enable_vary(tag='stage2')
opt._add_point_to_log()
assert opt.log()['target_active', -1] == 'yyyyyyyyyyyyyy'
assert opt.log()['vary_active', -1] == 'yyyyyyyyyyyyyyyyyyyy'

opt.solve()
opt._add_point_to_log()
assert opt.log()['target_active', -1] == 'yyyyyyyyyyyyyy'
assert opt.log()['vary_active', -1] == 'yyyyyyyyyyyyyyyyyyyy'
assert opt.log()['tol_met', -1] == 'yyyyyyyyyyyyyy'

assert np.isclose(tw.lhcb1['betx', 'ip1'], 0.15, atol=1e-6, rtol=0)
assert np.isclose(tw.lhcb1['bety', 'ip1'], 0.15, atol=1e-6, rtol=0)
assert np.isclose(tw.lhcb2['betx', 'ip1'], 0.15, atol=1e-6, rtol=0)
assert np.isclose(tw.lhcb2['bety', 'ip1'], 0.15, atol=1e-6, rtol=0)

assert np.isclose(tw.lhcb1['betx', 'ip5'], 0.15, atol=1e-6, rtol=0)
assert np.isclose(tw.lhcb1['bety', 'ip5'], 0.15, atol=1e-6, rtol=0)
assert np.isclose(tw.lhcb2['betx', 'ip5'], 0.15, atol=1e-6, rtol=0)
assert np.isclose(tw.lhcb2['bety', 'ip5'], 0.15, atol=1e-6, rtol=0)

assert np.isclose(tw.lhcb1['betx', 'ip8'], 1.5, atol=1e-6, rtol=0)
assert np.isclose(tw.lhcb1['bety', 'ip8'], 1.5, atol=1e-6, rtol=0)
assert np.isclose(tw.lhcb2['betx', 'ip8'], 1.5, atol=1e-6, rtol=0)
assert np.isclose(tw.lhcb2['bety', 'ip8'], 1.5, atol=1e-6, rtol=0)

assert np.isclose(tw.lhcb1['betx', 'ip2'], 10., atol=1e-5, rtol=0)
assert np.isclose(tw.lhcb1['bety', 'ip2'], 10., atol=1e-5, rtol=0)
assert np.isclose(tw.lhcb2['betx', 'ip2'], 10., atol=1e-5, rtol=0)
assert np.isclose(tw.lhcb2['bety', 'ip2'], 10., atol=1e-5, rtol=0)