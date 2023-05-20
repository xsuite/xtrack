import time

import xtrack as xt
import xpart as xp
import xdeps as xd

from cpymad.madx import Madx

# xt._print.suppress = True

# Load the line
line = xt.Line.from_json(
    '../../test_data/hllhc15_noerrors_nobb/line_w_knobs_and_particle.json')
line.particle_ref = xp.Particles(p0c=7e12, mass=xp.PROTON_MASS_EV)
collider = xt.Multiline(lines={'lhcb1': line})
collider.build_trackers()

tw_ref = collider.lhcb1.twiss()

ele_start_match = 's.ds.l7.b1'
ele_end_match = 'e.ds.r7.b1'
tw_init = tw_ref.get_twiss_init(ele_start_match)

betx_end_match = tw_ref['betx', ele_end_match]
bety_end_match = tw_ref['bety', ele_end_match]
alfx_end_match = tw_ref['alfx', ele_end_match]
alfy_end_match = tw_ref['alfy', ele_end_match]
dx_end_match = tw_ref['dx', ele_end_match]
dpx_end_match = tw_ref['dpx', ele_end_match]
mux_end_match = tw_ref['mux', ele_end_match]
muy_end_match = tw_ref['muy', ele_end_match]

betx_at_ip7 = tw_ref['betx', 'ip7']
bety_at_ip7 = tw_ref['bety', 'ip7']
alfx_at_ip7 = tw_ref['alfx', 'ip7']
alfy_at_ip7 = tw_ref['alfy', 'ip7']
dx_at_ip7 = tw_ref['dx', 'ip7']
dpx_at_ip7 = tw_ref['dpx', 'ip7']

scale = 23348.89927
scmin = 0.03*7000./line.vars['nrj']._value
qtlimitx28 = 1.0*225.0/scale
qtlimitx15 = 1.0*205.0/scale
qtlimit2 = 1.0*160.0/scale
qtlimit3 = 1.0*200.0/scale
qtlimit4 = 1.0*125.0/scale
qtlimit5 = 1.0*120.0/scale
qtlimit6 = 1.0*90.0/scale


# Break something

perturbed_vars = {}
perturbed_vars['kqt13.l7b1'] = collider.vars['kqt13.l7b1']._value * 1.1
perturbed_vars['kqt12.l7b1'] = collider.vars['kqt12.l7b1']._value * 1.1
perturbed_vars['kqtl11.l7b1'] = collider.vars['kqtl11.l7b1']._value * 0.9
perturbed_vars['kqtl10.l7b1'] = collider.vars['kqtl10.l7b1']._value * 1.1
perturbed_vars['kqtl9.l7b1'] = collider.vars['kqtl9.l7b1']._value * 0.8
perturbed_vars['kqtl8.l7b1'] = collider.vars['kqtl8.l7b1']._value * 0.9
perturbed_vars['kqtl7.l7b1'] = collider.vars['kqtl7.l7b1']._value * 1.12
perturbed_vars['kq6.l7b1'] = collider.vars['kq6.l7b1']._value * 0.9
perturbed_vars['kq6.r7b1'] = collider.vars['kq6.r7b1']._value * 1.1
perturbed_vars['kqtl7.r7b1'] = collider.vars['kqtl7.r7b1']._value * 1.1
perturbed_vars['kqtl8.r7b1'] = collider.vars['kqtl8.r7b1']._value * 0.85
perturbed_vars['kqtl9.r7b1'] = collider.vars['kqtl9.r7b1']._value * 0.95
perturbed_vars['kqtl10.r7b1'] = collider.vars['kqtl10.r7b1']._value * 0.85
perturbed_vars['kqtl11.r7b1'] = collider.vars['kqtl11.r7b1']._value * 1.15
perturbed_vars['kqt12.r7b1'] = collider.vars['kqt12.r7b1']._value * 0.9
perturbed_vars['kqt13.r7b1'] = collider.vars['kqt13.r7b1']._value * 1.12

tw_before = collider.lhcb1.twiss()

fact_on_tol = 0.001

action_tw_b1_ir7 = xt.match.ActionTwiss(line=collider.lhcb1,
        ele_start=ele_start_match, ele_stop=ele_end_match, twiss_init=tw_init)

for i_repeat in range(1):

    for nn, vv in perturbed_vars.items():
        collider.vars[nn]= vv

    t_start = time.perf_counter()
    opt = xd.Optimize(
        # verbose=True,
        targets=[
            xd.Target(action=action_tw_b1_ir7, tar=('dx',   'ip7'       ), value=dx_at_ip7, tol=1e-3*fact_on_tol),
            xd.Target(action=action_tw_b1_ir7, tar=('dpx',  'ip7'       ), value=dpx_at_ip7, tol=1e-5*fact_on_tol),
            xd.Target(action=action_tw_b1_ir7, tar=('betx', 'ip7'       ), value=betx_at_ip7, tol=1e-3*fact_on_tol),
            xd.Target(action=action_tw_b1_ir7, tar=('bety', 'ip7'       ), value=bety_at_ip7, tol=1e-3*fact_on_tol),
            xd.Target(action=action_tw_b1_ir7, tar=('alfx', 'ip7'       ), value=alfx_at_ip7, tol=1e-5*fact_on_tol),
            xd.Target(action=action_tw_b1_ir7, tar=('alfy', 'ip7'       ), value=alfy_at_ip7, tol=1e-5*fact_on_tol),
            xd.Target(action=action_tw_b1_ir7, tar=('alfx', 'e.ds.r7.b1'), value=alfx_end_match, tol=1e-5*fact_on_tol),
            xd.Target(action=action_tw_b1_ir7, tar=('alfy', 'e.ds.r7.b1'), value=alfy_end_match, tol=1e-5*fact_on_tol),
            xd.Target(action=action_tw_b1_ir7, tar=('betx', 'e.ds.r7.b1'), value=betx_end_match, tol=1e-3*fact_on_tol),
            xd.Target(action=action_tw_b1_ir7, tar=('bety', 'e.ds.r7.b1'), value=bety_end_match, tol=1e-3*fact_on_tol),
            xd.Target(action=action_tw_b1_ir7, tar=('dx',   'e.ds.r7.b1'), value=dx_end_match, tol=1e-3*fact_on_tol),
            xd.Target(action=action_tw_b1_ir7, tar=('dpx',  'e.ds.r7.b1'), value=dpx_end_match, tol=1e-5*fact_on_tol),
            xd.Target(action=action_tw_b1_ir7, tar=('mux',  'e.ds.r7.b1'), value=mux_end_match, tol=1e-5*fact_on_tol),
            xd.Target(action=action_tw_b1_ir7, tar=('muy', 'e.ds.r7.b1'), value=muy_end_match, tol=1e-5*fact_on_tol),
        ],
        vary=[
            xd.Vary('kqt13.l7b1',  collider.vars, step=1.0E-9, limits=(-qtlimit5, qtlimit5)),
            xd.Vary('kqt12.l7b1',  collider.vars, step=1.0E-9, limits=(-qtlimit5, qtlimit5)),
            xd.Vary('kqtl11.l7b1', collider.vars, step=1.0E-9, limits=(-qtlimit4*300./550., qtlimit4*300./550.)),
            xd.Vary('kqtl10.l7b1', collider.vars, step=1.0E-9, limits=(-qtlimit4*500./550., qtlimit4*500./550.)),
            xd.Vary('kqtl9.l7b1',  collider.vars, step=1.0E-9, limits=(-qtlimit4*400./550., qtlimit4*400./550.)),
            xd.Vary('kqtl8.l7b1',  collider.vars, step=1.0E-9, limits=(-qtlimit4*300./550., qtlimit4*300./550.)),
            xd.Vary('kqtl7.l7b1',  collider.vars, step=1.0E-9, limits=(-qtlimit4, qtlimit4)),
            xd.Vary('kq6.l7b1',    collider.vars, step=1.0E-9, limits=(-qtlimit6, qtlimit6)),
            xd.Vary('kq6.r7b1',    collider.vars, step=1.0E-9, limits=(-qtlimit6, qtlimit6)),
            xd.Vary('kqtl7.r7b1',  collider.vars, step=1.0E-9, limits=(-qtlimit4, qtlimit4)),
            xd.Vary('kqtl8.r7b1',  collider.vars, step=1.0E-9, limits=(-qtlimit4*550./550., qtlimit4*550./550.)),
            xd.Vary('kqtl9.r7b1',  collider.vars, step=1.0E-9, limits=(-qtlimit4*500./550., qtlimit4*500./550.)),
            xd.Vary('kqtl10.r7b1', collider.vars, step=1.0E-9, limits=(-qtlimit4, qtlimit4)),
            xd.Vary('kqtl11.r7b1', collider.vars, step=1.0E-9, limits=(-qtlimit4, qtlimit4)),
            xd.Vary('kqt12.r7b1',  collider.vars, step=1.0E-9, limits=(-qtlimit5, qtlimit5)),
            xd.Vary('kqt13.r7b1',  collider.vars, step=1.0E-9, limits=(-qtlimit5, qtlimit5)),
        ]
    )
    match_res = opt.solve()
    t_end = time.perf_counter()
    print(f"Matching time: {t_end - t_start:0.4f} seconds")


tw_after = collider.lhcb1.twiss()