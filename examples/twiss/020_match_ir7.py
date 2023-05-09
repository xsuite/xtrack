import time

import xtrack as xt
import xpart as xp

from cpymad.madx import Madx

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

for i_repeat in range(1):

    for nn, vv in perturbed_vars.items():
        collider.vars[nn]= vv

    t_start = time.perf_counter()
    match_res = collider.match(
        #verbose=True,
        ele_start=ele_start_match,
        ele_stop=ele_end_match,
        twiss_init=tw_init,
        targets=[
            xt.Target(line='lhcb1', at='ip7',        tar='dx',   value=dx_at_ip7, tol=1e-3),
            xt.Target(line='lhcb1', at='ip7',        tar='dpx',  value=dpx_at_ip7, tol=1e-5),
            xt.Target(line='lhcb1', at='ip7',        tar='betx', value=betx_at_ip7, tol=1e-3),
            xt.Target(line='lhcb1', at='ip7',        tar='bety', value=bety_at_ip7, tol=1e-3),
            xt.Target(line='lhcb1', at='ip7',        tar='alfx', value=alfx_at_ip7, tol=1e-5),
            xt.Target(line='lhcb1', at='ip7',        tar='alfy', value=alfy_at_ip7, tol=1e-5),
            xt.Target(line='lhcb1', at='e.ds.r7.b1', tar='alfx', value=alfx_end_match, tol=1e-5),
            xt.Target(line='lhcb1', at='e.ds.r7.b1', tar='alfy', value=alfy_end_match, tol=1e-5),
            xt.Target(line='lhcb1', at='e.ds.r7.b1', tar='betx', value=betx_end_match, tol=1e-3),
            xt.Target(line='lhcb1', at='e.ds.r7.b1', tar='bety', value=bety_end_match, tol=1e-3),
            xt.Target(line='lhcb1', at='e.ds.r7.b1', tar='dx',   value=dx_end_match, tol=1e-3),
            xt.Target(line='lhcb1', at='e.ds.r7.b1', tar='dpx',  value=dpx_end_match, tol=1e-5),
            xt.Target(line='lhcb1', at='e.ds.r7.b1', tar='mux',  value=mux_end_match, tol=1e-5),
            xt.Target(line='lhcb1', at='e.ds.r7.b1', tar='muy ', value=muy_end_match, tol=1e-5),
            # xt.TargetInequality('bety', '<', 180.49-0.3, line='lhcb1', at='mq.11l7.b1'),
            # xt.TargetInequality('bety', '<', 174.5,      line='lhcb1', at='mq.9l7.b1'),
            # xt.TargetInequality('bety', '<', 176.92,     line='lhcb1', at='mq.8r7.b1'),
            # xt.TargetInequality('bety', '<', 179,        line='lhcb1', at='mq.10r7.b1'),
        ],
        vary=[
            xt.Vary('kqt13.l7b1',  step=1.0E-9, limits=(-qtlimit5, qtlimit5)),
            xt.Vary('kqt12.l7b1',  step=1.0E-9, limits=(-qtlimit5, qtlimit5)),
            xt.Vary('kqtl11.l7b1', step=1.0E-9, limits=(-qtlimit4*300./550., qtlimit4*300./550.)),
            xt.Vary('kqtl10.l7b1', step=1.0E-9, limits=(-qtlimit4*500./550., qtlimit4*500./550.)),
            xt.Vary('kqtl9.l7b1',  step=1.0E-9, limits=(-qtlimit4*400./550., qtlimit4*400./550.)),
            xt.Vary('kqtl8.l7b1',  step=1.0E-9, limits=(-qtlimit4*300./550., qtlimit4*300./550.)),
            xt.Vary('kqtl7.l7b1',  step=1.0E-9, limits=(-qtlimit4, qtlimit4)),
            xt.Vary('kq6.l7b1',    step=1.0E-9, limits=(-qtlimit6, qtlimit6)),
            xt.Vary('kq6.r7b1',    step=1.0E-9, limits=(-qtlimit6, qtlimit6)),
            xt.Vary('kqtl7.r7b1',  step=1.0E-9, limits=(-qtlimit4, qtlimit4)),
            xt.Vary('kqtl8.r7b1',  step=1.0E-9, limits=(-qtlimit4*550./550., qtlimit4*550./550.)),
            xt.Vary('kqtl9.r7b1',  step=1.0E-9, limits=(-qtlimit4*500./550., qtlimit4*500./550.)),
            xt.Vary('kqtl10.r7b1', step=1.0E-9, limits=(-qtlimit4, qtlimit4)),
            xt.Vary('kqtl11.r7b1', step=1.0E-9, limits=(-qtlimit4, qtlimit4)),
            xt.Vary('kqt12.r7b1',  step=1.0E-9, limits=(-qtlimit5, qtlimit5)),
            xt.Vary('kqt13.r7b1',  step=1.0E-9, limits=(-qtlimit5, qtlimit5)),
        ]
    )
    t_end = time.perf_counter()
    print(f"Matching time: {t_end - t_start:0.4f} seconds")


tw_after = collider.lhcb1.twiss()


_err = match_res['jac_solver'].func
x_final = match_res['res']

n_repeat_err_call = 100
t0 = time.perf_counter()
for _ in range(n_repeat_err_call):
    _err(x_final)
t1 = time.perf_counter()
print(f"Error call time: {1000 * (t1 - t0)/n_repeat_err_call:0.4f} ms")

n_repeat_set_knobs = 100
t0 = time.perf_counter()
for _ in range(n_repeat_set_knobs):
    knob_values = _err._x_to_knobs(x_final)
    for kk, vv in zip(_err.vary, knob_values):
        _err.line.vars[kk.name] = vv
t1 = time.perf_counter()
print(f"Set knobs time: {1000 * (t1 - t0)/n_repeat_set_knobs:0.4f} ms")

tw = _err._last_twiss
n_repeat_eval_targets = 100
t0 = time.perf_counter()
for _ in range(n_repeat_eval_targets):
    for tt in _err.targets:
        tt.eval(tw)
t1 = time.perf_counter()
print(f"Evaluate targets time: {1000 * (t1 - t0)/n_repeat_eval_targets:0.4f} ms")


tw_init = tw_ref.get_twiss_init(ele_start_match)
ele_index_start = line.element_names.index(ele_start_match)
ele_index_end = line.element_names.index(ele_end_match)

ttt = collider.twiss(
        ele_start=[ele_index_start],
        ele_stop=[ele_index_end],
        twiss_init=tw_init,
        _keep_initial_particles=True
        )

n_repeat_twiss = 100
t0 = time.perf_counter()
for _ in range(n_repeat_twiss):
    collider.twiss(
        ele_start=[ele_index_start],
        ele_stop=[ele_index_end],
        twiss_init=tw_init,
        _initial_particles=[ttt.lhcb1._initial_particles]
        )
t1 = time.perf_counter()
print(f"Twiss time: {1000 * (t1 - t0)/n_repeat_twiss:0.4f} ms")