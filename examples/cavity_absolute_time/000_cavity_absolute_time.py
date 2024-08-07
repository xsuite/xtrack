import numpy as np
from scipy.constants import c as clight

import xtrack as xt
import xpart as xp
import xobjects as xo
from functools import partial

line = xt.Line.from_json(
    '../../test_data/hllhc14_no_errors_with_coupling_knobs/line_b1.json')
# line.cycle('ip1')
line.build_tracker()

for vv in line.vars.get_table().rows[
    'on_x.*|on_sep.*|on_crab.*|on_alice|on_lhcb|corr_.*'].name:
    line.vars[vv] = 0

tw = line.twiss()

df_hz = -50
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


def merit_function(x, num_turns=10):
    p = line.build_particles(x=x[0], px=x[1], y=x[2], py=x[3], zeta=x[4], delta=x[5])
    line.track(p, num_turns=num_turns, turn_by_turn_monitor=True)
    rec = line.record_last_track
    dx = rec.x[0, :] - rec.x[0, 0]
    dpx = rec.px[0, :] - rec.px[0, 0]
    dy = rec.y[0, :] - rec.y[0, 0]
    dpy = rec.py[0, :] - rec.py[0, 0]
    ddelta = rec.delta[0, :] - rec.delta[0, 0]

    out = np.array(list(dx) + list(dpx) + list(dy) + list(dpy) + list(ddelta))
    # print(x, out)
    return out

num_turns = 20
opt = xt.match.opt_from_callable(partial(merit_function, num_turns=num_turns),
                           x0=np.array(6*[0.]),
                           steps=[1e-9, 1e-10, 1e-9, 1e-10, 1e-4, 1e-7],
                           tar=np.array(5*num_turns*[0.]),
                           tols=np.array(5*num_turns*[1e-10]))
# opt.targets[-1].weight = 1e4
opt.step(5)


x_sol = opt.get_knob_values()
particle_on_co = line.build_particles(
    x=x_sol[0], px=x_sol[1], y=x_sol[2], py=x_sol[3], zeta=x_sol[4],
    delta=x_sol[5])

tw1 = line.twiss(particle_on_co=particle_on_co)

T_rev = tw1.T_rev0 - (tw1.zeta[-1] - tw1.zeta[0])/(tw.beta0*clight)

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
ax1 = plt.subplot(2, 1, 1)
plt.plot(tw1.s, tw1.x*1e3, label='x')
plt.ylabel('x [mm]')
plt.grid()
ax2 = plt.subplot(2, 1, 2, sharex=ax1)
plt.plot(tw1.s, tw1.delta*1e3, label='delta')
plt.xlabel('s [m]')
plt.ylabel(r'$\delta$ [$10^{-3}$]')
plt.ylim(-0.5, 0.5)
plt.grid()

plt.suptitle(r'$\Delta f_{\mathrm{rf}}$ = ' +f'{df_hz} Hz, '
             f'expexted $\delta$ = {delta_expected*1e3:.2f}e-3')

plt.show()