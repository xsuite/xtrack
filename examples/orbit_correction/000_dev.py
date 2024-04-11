import xtrack as xt
import numpy as np

line = xt.Line.from_json(
    '../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')
tt = line.get_table()

tw0 = line.twiss4d()

# line['mq.14r7.b1'].shift_x=0.5e-3
# line['mq.14r2.b1'].shift_y=0.5e-3

tw1 = line.twiss4d()

# Select monitors by names (starting by "bpm." and not ending by "_entry" or "_exit")
tt_monitors = tt.rows['bpm\..*'].rows['.*(?<!_entry)$'].rows['.*(?<!_exit)$']
monitor_names = tt_monitors.name

# Select h correctors by names (starting by "mcb.", containing "h.", and ending by ".b1")
tt_h_correctors = tt.rows['mcb.*'].rows['.*h\..*'].rows['.*\.b1']
h_corrector_names = tt_h_correctors.name

# Select v correctors by names (starting by "mcb.", containing "v.", and ending by ".b1")
v_corrector_names = tt.rows['mcb.*'].rows['.*v\..*'].rows['.*\.b1'].name

betx_monitors = tw1.rows[monitor_names].betx
betx_hcorrectors = tw1.rows[h_corrector_names].betx

mux_monitor = tw1.rows[monitor_names].mux
mux_hcorrectors = tw1.rows[h_corrector_names].mux

n_monitors = len(monitor_names)
n_hcorrectors = len(h_corrector_names)

qx = tw1.qx

from numpy.matlib import repmat

# Slide 28
# https://indico.cern.ch/event/1328128/contributions/5589794/attachments/2786478/4858384/linearimperfections_2024.pdf

bet_prod = np.atleast_2d(betx_monitors).T @ np.atleast_2d(betx_hcorrectors)
mux_diff = repmat(mux_monitor, n_hcorrectors, 1).T - repmat(mux_hcorrectors, n_monitors, 1)

response_matrix = np.sqrt(bet_prod) / 2 / np.sin(np.pi * qx) * np.cos(np.pi * qx - 2*np.pi*np.abs(mux_diff))

theta = 1e-5
i_h_kick = np.where(h_corrector_names == 'mcbh.15r7.b1')[0][0]
line['mcbh.15r7.b1'].knl[0] = theta

tw2 = line.twiss4d()

kick_vect = np.zeros(n_hcorrectors)
kick_vect[i_h_kick] = -theta

x_res = response_matrix @ kick_vect

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.plot(tt_monitors.s, x_res, '.', label='Response')
plt.plot(tw2.s, tw2.x)

plt.show()

