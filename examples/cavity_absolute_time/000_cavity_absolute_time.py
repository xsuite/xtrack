import numpy as np

import xtrack as xt
import xpart as xp
import xobjects as xo

line = xt.Line.from_json(
    '../../test_data/hllhc14_no_errors_with_coupling_knobs/line_b1.json')
line.cycle('ip1')
line.build_tracker()

for vv in line.vars.get_table().rows[
    'on_x.*|on_sep.*|on_crab.*|on_alice|on_lhcb|corr_.*'].name:
    line.vars[vv] = 0

tw = line.twiss()

df_hz = 1
h_rf = 35640
f_rev = 1/tw.T_rev0
df_rev = df_hz / h_rf
eta = tw.slip_factor
delta_expected = -df_rev / f_rev / eta

line.vars['f_rf'] = 400789598.9858259 + df_hz
tt = line.get_table()
for nn in tt.rows[tt.element_type=='Cavity'].name:
    line.element_refs[nn].absolute_time = 1
    line.element_refs[nn].frequency = line.vars['f_rf']

line.particle_ref.t_sim = tw.T_rev0


# p = line.build_particles(delta=np.linspace(delta_expected-1e-5, delta_expected+1e-5, 111))
p = line.build_particles(delta=delta_expected,
                         zeta=np.linspace(-0.04 ,0.04, 200))
p.t_sim = line.particle_ref.t_sim
line.track(p, num_turns=100, with_progress=True, turn_by_turn_monitor=True)
rec = line.record_last_track

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.plot(rec.zeta.T, rec.delta.T)
plt.axhline(delta_expected, color='C1', linestyle='--', label='expected')

plt.show()

# def merit_function(x):
#     p = line.build_particles(x=x[0], px=x[1], y=x[2], py=x[3], zeta=x[4], delta=x[5])
#     p.t_sim = line.particle_ref.t_sim
#     line.track(p, num_turns=10, turn_by_turn_monitor=True)
#     rec = line.record_last_track
#     dx = rec.x[0, -1] - rec.x[0, 0]
#     dpx = rec.px[0, -1] - rec.px[0, 0]
#     dy = rec.y[0, -1] - rec.y[0, 0]
#     dpy = rec.py[0, -1] - rec.py[0, 0]
#     ddelta = rec.delta[0, -1] - rec.delta[0, 0]
#     delta_rms = 100*np.std(rec.delta[0, :])

#     out = np.array([dx, dpx, dy, dpy, delta_rms])
#     return out


# opt = xt.match.opt_from_callable(merit_function, np.array(6*[0]),
#                            steps=[1e-6, 1e-7, 1e-6, 1e-7, 1e-3, 1e-5],
#                            tar=np.array(5*[0]),
#                            tols=[1e-8, 1e-10, 1e-8, 1e-10, 1e-10])


def merit_function(x):
    p = line.build_particles(zeta=x[0], delta=x[1])
    p.t_sim = line.particle_ref.t_sim
    line.track(p, num_turns=10, turn_by_turn_monitor=True)
    rec = line.record_last_track
    dx = rec.x[0, -1] - rec.x[0, 0]
    dpx = rec.px[0, -1] - rec.px[0, 0]
    dy = rec.y[0, -1] - rec.y[0, 0]
    dpy = rec.py[0, -1] - rec.py[0, 0]
    ddelta = rec.delta[0, -1] - rec.delta[0, 0]
    delta_rms = 100*np.std(rec.delta[0, :])

    out = np.array([delta_rms])
    print(x, out)
    return out


opt = xt.match.opt_from_callable(merit_function, np.array(2*[0.]),
                           steps=[1e-3, 1e-7],
                           tar=np.array([0]),
                           tols=np.array([1e-7]))

opt.solve()