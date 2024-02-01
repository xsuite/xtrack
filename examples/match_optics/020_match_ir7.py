import time

import xtrack as xt

from cpymad.madx import Madx

# xt._print.suppress = True

# Load the line
line = xt.Line.from_json(
    '../../test_data/hllhc15_noerrors_nobb/line_w_knobs_and_particle.json')
line.particle_ref = xt.Particles(p0c=7e12, mass=xt.PROTON_MASS_EV)
collider = xt.Multiline(lines={'lhcb1': line})
collider.build_trackers()
collider.vars.cache_active = True

scale = 23348.89927
scmin = 0.03*7000./line.vars['nrj']._value
qtlimitx28 = 1.0*225.0/scale
qtlimitx15 = 1.0*205.0/scale
qtlimit2 = 1.0*160.0/scale
qtlimit3 = 1.0*200.0/scale
qtlimit4 = 1.0*125.0/scale
qtlimit5 = 1.0*120.0/scale
qtlimit6 = 1.0*90.0/scale

collider.vars.vary_default.update({
    'kqt13.l7b1':  {'step': 1.0E-9, 'limits': (-qtlimit5, qtlimit5)},
    'kqt12.l7b1':  {'step': 1.0E-9, 'limits': (-qtlimit5, qtlimit5)},
    'kqtl11.l7b1': {'step': 1.0E-9, 'limits': (-qtlimit4*300./550., qtlimit4*300./550.)},
    'kqtl10.l7b1': {'step': 1.0E-9, 'limits': (-qtlimit4*500./550., qtlimit4*500./550.)},
    'kqtl9.l7b1':  {'step': 1.0E-9, 'limits': (-qtlimit4*400./550., qtlimit4*400./550.)},
    'kqtl8.l7b1':  {'step': 1.0E-9, 'limits': (-qtlimit4*300./550., qtlimit4*300./550.)},
    'kqtl7.l7b1':  {'step': 1.0E-9, 'limits': (-qtlimit4, qtlimit4)},
    'kq6.l7b1':    {'step': 1.0E-9, 'limits': (-qtlimit6, qtlimit6)},
    'kq6.r7b1':    {'step': 1.0E-9, 'limits': (-qtlimit6, qtlimit6)},
    'kqtl7.r7b1':  {'step': 1.0E-9, 'limits': (-qtlimit4, qtlimit4)},
    'kqtl8.r7b1':  {'step': 1.0E-9, 'limits': (-qtlimit4*550./550., qtlimit4*550./550.)},
    'kqtl9.r7b1':  {'step': 1.0E-9, 'limits': (-qtlimit4*500./550., qtlimit4*500./550.)},
    'kqtl10.r7b1': {'step': 1.0E-9, 'limits': (-qtlimit4, qtlimit4)},
    'kqtl11.r7b1': {'step': 1.0E-9, 'limits': (-qtlimit4, qtlimit4)},
    'kqt12.r7b1':  {'step': 1.0E-9, 'limits': (-qtlimit5, qtlimit5)},
    'kqt13.r7b1':  {'step': 1.0E-9, 'limits': (-qtlimit5, qtlimit5)},
})
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
    tw0 = collider.twiss()
    opt = collider.match(
        solve=False,
        start=ele_start_match,
        end=ele_end_match,
        init=tw_ref, init_at=xt.START,
        targets=[
            xt.TargetList(
                ('betx', 'bety', 'alfx', 'alfy', 'dx', 'dpx'),
                at='ip7', line='lhcb1', value=tw_ref),
            xt.TargetList(
                ('alfx', 'alfy', 'betx', 'bety', 'dx', 'dpx', 'mux', 'muy'),
                at='e.ds.r7.b1',line='lhcb1', value=tw_ref),
            ],
        vary=[
            xt.VaryList(
                ('kqt13.l7b1', 'kqt12.l7b1', 'kqtl11.l7b1', 'kqtl10.l7b1',
                 'kqtl9.l7b1', 'kqtl8.l7b1', 'kqtl7.l7b1', 'kq6.l7b1',
                 'kq6.r7b1', 'kqtl7.r7b1', 'kqtl8.r7b1', 'kqtl9.r7b1',
                 'kqtl10.r7b1', 'kqtl11.r7b1', 'kqt12.r7b1', 'kqt13.r7b1'))
        ]
    )

    match_res = opt.solve()
    t_end = time.perf_counter()
    print(f"Matching time: {t_end - t_start:0.4f} seconds")


tw_after = collider.lhcb1.twiss()


_err = opt._err
x_final = opt.solver._xbest

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
        _err.actions[0].line.vars[kk.name] = vv
t1 = time.perf_counter()
print(f"Set knobs time: {1000 * (t1 - t0)/n_repeat_set_knobs:0.4f} ms")

last_data = _err._last_data
n_repeat_eval_targets = 100
t0 = time.perf_counter()
for _ in range(n_repeat_eval_targets):
    for tt in _err.targets:
        tt.eval(last_data)
t1 = time.perf_counter()
print(f"Evaluate targets time: {1000 * (t1 - t0)/n_repeat_eval_targets:0.4f} ms")


tw_init = tw_ref.get_twiss_init(ele_start_match)
ele_index_start = line.element_names.index(ele_start_match)
ele_index_end = line.element_names.index(ele_end_match)

ttt = collider.twiss(
        start=[ele_index_start],
        end=[ele_index_end],
        init=tw_init,
        _keep_initial_particles=True,
        _keep_tracking_data=True,
        )

n_repeat_twiss = 100
t0 = time.perf_counter()
for _ in range(n_repeat_twiss):
    collider.twiss(
        start=[ele_index_start],
        end=[ele_index_end],
        init=tw_init,
        _ebe_monitor=[ttt.lhcb1.tracking_data],
        _initial_particles=[ttt.lhcb1._initial_particles]
        )
t1 = time.perf_counter()
print(f"Twiss time: {1000 * (t1 - t0)/n_repeat_twiss:0.4f} ms")

n_repeat_tracking = 100
p_test = [ttt.lhcb1._initial_particles.copy() for _ in range(n_repeat_tracking)]
t0 = time.perf_counter()
for ii in range(n_repeat_tracking):
    collider.lhcb1.track(
        p_test[ii],
        start=ele_index_start,
        end=ele_index_end,
        turn_by_turn_monitor=ttt.lhcb1.tracking_data
        )
t1 = time.perf_counter()
print(f"Tracking time: {1000 * (t1 - t0)/n_repeat_tracking:0.4f} ms")
