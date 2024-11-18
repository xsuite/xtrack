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
line['acta.31637'].voltage = 1.2e6 * line.ref['on_rf1']
line['acta.31637'].lag = 180

line['actd.31934'].absolute_time = 1
line['actd.31934'].frequency = h_rf * f0 - line.ref['dfreq']
line['actd.31934'].voltage = 1.2e6 * line.ref['on_rf2']
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
# bunch0 = xp.generate_matched_gaussian_bunch(sigma_z=0.15, num_particles=500,
#                                             nemitt_x=0, nemitt_y=0, line=line)
bucket_length = line.get_length() / h_rf
train0 = xp.generate_matched_gaussian_multibunch_beam(
    sigma_z=0.1, nemitt_x=0, nemitt_y=0, line=line,
    bunch_intensity_particles=1e11,
    bunch_num_particles=500, filling_scheme=5*[1, 0, 0, 0, 0])
line['on_rf2'] = 1.
line['dfreq'] = dfreq_tmp

train1 = train0.copy()
train1.delta += tw_rf1.delta[0]
train2 = train0.copy()
train2.delta += tw_rf2.delta[0]
train2.zeta += 20 * bucket_length

two_trains = xt.Particles.merge([train1, train2])

num_turns = 400
line.track(two_trains, num_turns=num_turns, with_progress=10,
           turn_by_turn_monitor=True)
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
plt.plot(two_trains.zeta, two_trains.delta, '.')

i_turn = 350
plt.figure(4)
plt.plot(mon.zeta.T[i_turn, :], mon.delta.T[i_turn, :], '.')

plt.show()

# def update_plot(i_turn):
#     plt.clf()
#     plt.plot(mon.zeta.T[i_turn, :], mon.delta.T[i_turn, :], '.', markersize=1)
#     plt.xlim(-60, 60)
#     plt.ylim(-12e-3, 12e-3)
#     plt.xlabel('z [m]')
#     plt.ylabel(r'$\Delta p / p_0$')
#     plt.title(f'Turn {i_turn}')
#     plt.subplots_adjust(left=0.2)
#     plt.grid(alpha=0.5)


# import matplotlib.animation as animation
# fig = plt.figure()
# animation_fig = animation.FuncAnimation(
#     fig, update_plot, frames=range(0, num_turns, 3))
# animation_fig.save('slipstack.gif', fps=10)

def update_plot(i_turn, fig):
    plt.clf()
    plt.plot(mon.zeta.T[i_turn, :], mon.delta.T[i_turn, :], '.', markersize=1)
    plt.xlim(-40, 40)
    plt.ylim(-12e-3, 12e-3)
    plt.xlabel('z [m]')
    plt.ylabel(r'$\Delta p / p_0$')
    plt.title(f'Turn {i_turn}')
    plt.subplots_adjust(left=0.2)
    plt.grid(alpha=0.5)

fig = plt.figure()
from matplotlib.animation import FFMpegFileWriter
moviewriter = FFMpegFileWriter(fps=15)
with moviewriter.saving(fig, 'slipstack.mp4', dpi=100):
    for j in range(0, num_turns, 2):
        update_plot(j, fig)
        moviewriter.grab_frame()



# def update_plot(i_turn):
#     plt.clf()
#     plt.plot(mon.zeta.T[i_turn, :], mon.y.T[i_turn, :], '.', markersize=1)
#     plt.xlim(-60, 60)
#     plt.ylim(-5e-3, 5e-3)
#     plt.xlabel('z [m]')
#     plt.ylabel(r'y [mm]')
#     plt.title(f'Turn {i_turn}')
#     plt.subplots_adjust(left=0.2)
#     plt.grid(alpha=0.5)

# import matplotlib.animation as animation
# fig = plt.figure()
# animation_fig = animation.FuncAnimation(
#     fig, update_plot, frames=range(0, num_turns, 1))
# animation_fig.save('slipstack_y.gif', fps=30)

