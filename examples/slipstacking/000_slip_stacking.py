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

# Remove existing voltage from all cavities
tt = line.get_table(attr=True)
tt_cav = tt.rows[tt.element_type=='Cavity']
for nn in tt_cav.name:
    line[nn].voltage = 0

# Define knobs controlling the RF cavities
line['on_dfreq'] = 1.
line['on_rf1'] = 1.
line['on_rf2'] = 1.
line['dfreq'] = 1.e3 * line.ref['on_dfreq'] # Frequency shift in Hz
line['v_rf1'] = 1.2e6 * line.ref['on_rf1']
line['v_rf2'] = 1.2e6 * line.ref['on_rf2']

# Setup two RF cavities with different frequencies
line['acta.31637'].absolute_time = 1 # <-- define cavity w.r.t. absolute time and not to ref. particle!!!
line['acta.31637'].frequency = h_rf * f0 + line.ref['dfreq']
line['acta.31637'].voltage = line.ref['v_rf1']
line['acta.31637'].lag = 180

line['actd.31934'].absolute_time = 1 # <-- define cavity w.r.t. absolute time and not to ref. particle!!!
line['actd.31934'].frequency = h_rf * f0 - line.ref['dfreq']
line['actd.31934'].voltage = line.ref['v_rf2']
line['actd.31934'].lag = 180

# Display the cavity properties
tt = line.get_table(attr=True)
tt_cav = tt.rows[['acta.31637', 'actd.31934']]
tt_cav.cols['frequency voltage lag'].show(digits=10)

# Twiss (looking for actual revolution frequency)
line['on_rf2'] = 0.
tw_rf1 = line.twiss(search_for_t_rev=True)
line['on_rf2'] = 1.

line['on_rf1'] = 0.
tw_rf2 = line.twiss(search_for_t_rev=True)
line['on_rf1'] = 1.

print('Revolution frequency with RF1 only: ', 1/tw_rf1.T_rev)
print('Revolution frequency with RF2 only: ', 1/tw_rf2.T_rev)
print('Energy deviation with RF1 only: ', tw_rf1.ptau[0])
print('Energy deviation with RF2 only: ', tw_rf2.ptau[0])

# match a train (no frequency shift and single RF)
dfreq_tmp = line['dfreq']
line['dfreq'] = 0
line['on_rf2'] = 0.
import xpart as xp
train0 = xp.generate_matched_gaussian_multibunch_beam(
    sigma_z=0.1, nemitt_x=0, nemitt_y=0, line=line,
    bunch_intensity_particles=1e11,
    bunch_num_particles=500, filling_scheme=5*[1, 0, 0, 0, 0])
line['on_rf2'] = 1.
line['dfreq'] = dfreq_tmp

# Make two trains matched to the two RF systems
train1 = train0.copy()
train1.delta += tw_rf1.delta[0]
train2 = train0.copy()
train2.delta += tw_rf2.delta[0]
bucket_length = line.get_length() / h_rf
train2.zeta += 20 * bucket_length

# Make a single particle set with the two trains
two_trains = xt.Particles.merge([train1, train2])

# Track!
num_turns = 400
line.track(two_trains, num_turns=num_turns, with_progress=10,
           turn_by_turn_monitor=True)
mon = line.record_last_track

# Plots
import matplotlib.pyplot as plt
plt.close('all')

# Plot one turn
i_turn = 350
plt.figure(4)
plt.plot(mon.zeta.T[i_turn, :], mon.delta.T[i_turn, :], '.')
plt.show()

# Make movie (needed `conda install -c conda-forge ffmpeg``)
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
