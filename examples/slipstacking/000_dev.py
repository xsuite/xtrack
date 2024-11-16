# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np
import xtrack as xt
from scipy.constants import c as clight

line = xt.Line.from_json('../../test_data/sps_w_spacecharge/line_no_spacecharge.json')
line.build_tracker()
tw0 = line.twiss4d()

f0 = 1 / (line.get_length()/(clight*line.particle_ref.beta0[0]))
h_rf = 4620

tt = line.get_table(attr=True)
tt_cav = tt.rows[tt.element_type=='Cavity']
tt_cav.cols['frequency voltage lag'].show()

line['dfreq'] = 1.e3
line['on_rf1'] = 1.
line['on_rf2'] = 1.

line['acta.31637'].absolute_time = 1
line['acta.31637'].frequency = h_rf * f0 + line.ref['dfreq']
line['acta.31637'].voltage = 1.5e6 * line.ref['on_rf1']
line['acta.31637'].lag = 180

line['actd.31934'].absolute_time = 1
line['actd.31934'].frequency = h_rf * f0 - line.ref['dfreq']
line['actd.31934'].voltage = 1.5e6 * line.ref['on_rf2']
line['actd.31934'].lag = 180

tt = line.get_table(attr=True)
tt_cav = tt.rows[tt.element_type=='Cavity']
tt_cav.cols['frequency voltage lag'].show()

line['on_rf2'] = 0.
tw_rf1 = line.twiss(search_for_t_rev=True)
line['on_rf2'] = 1.

line['on_rf1'] = 0.
tw_rf2 = line.twiss(search_for_t_rev=True)
line['on_rf1'] = 1.


# particles = xt.Particles(p0c=26e9, zeta=np.linspace(-1, 1, 40), delta=tw.delta[0])
particles = line.build_particles(
    method='4d',
    x_norm=0, y_norm=0,
    delta=np.linspace(-8e-3, 8e-3, 1000))

particles.t_sim = line.get_length() / line.particle_ref._xobject.beta0[0] / clight
line.track(particles, num_turns=1000, turn_by_turn_monitor=True, with_progress=10)
rec = line.record_last_track

# match a bunch (no frequency shift)
dfreq_tmp = line['dfreq']
line['dfreq'] = 0
line['on_rf2'] = 0.
import xpart as xp
bunch0 = xp.generate_matched_gaussian_bunch(sigma_z=0.15, num_particles=500,
                                            nemitt_x=0, nemitt_y=0, line=line)
line['on_rf2'] = 1.
line['dfreq'] = dfreq_tmp

bunch1 = bunch0.copy()
bunch1.delta += tw_rf1.delta[0]
bunch2 = bunch0.copy()
bunch2.delta += tw_rf2.delta[0]

two_bunches = xt.Particles.merge([bunch1, bunch2])
line.track(two_bunches, num_turns=1000, with_progress=10, turn_by_turn_monitor=True)

mon = line.record_last_track

import matplotlib.pyplot as plt


plt.close('all')
plt.figure(1)
for ii in range(rec.x.shape[0]):
    mask = rec.state[ii, :]>0
    plt.plot(rec.zeta[ii, mask], rec.delta[ii, mask])

plt.grid(linestyle='--')
plt.xlabel('z [m]')
plt.ylabel(r'$\Delta p / p_0$')

plt.figure(2)
plt.plot(rec.zeta.T)

plt.figure(3)
plt.plot(two_bunches.zeta, two_bunches.delta, '.')

i_turn = 500
plt.figure(4)
plt.plot(mon.zeta.T[i_turn, :], mon.delta.T[i_turn, :], '.')

def update_plot(i_turn):
    plt.clf()
    plt.plot(mon.zeta.T[i_turn, :], mon.delta.T[i_turn, :], '.', markersize=1)
    plt.xlim(-50, 50)
    plt.ylim(-5e-3, 5e-3)
    plt.xlabel('z [m]')
    plt.ylabel(r'$\Delta p / p_0$')
    plt.title(f'Turn {i_turn}')

import matplotlib.animation as animation
fig = plt.figure()
animation_fig = animation.FuncAnimation(fig, update_plot, frames=range(0, 1000, 5))
# animation_fig.save('slipstack.gif', fps=30)

plt.show()
