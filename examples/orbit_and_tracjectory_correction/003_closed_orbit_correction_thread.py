import xtrack as xt
import numpy as np
from numpy.matlib import repmat

import xtrack.trajectory_correction as oc


line = xt.Line.from_json(
    '../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')
tt = line.get_table()

# Define elements to be used as monitors for orbit correction
# (for LHC all element names starting by "bpm" and not ending by "_entry" or "_exit")
tt_monitors = tt.rows['bpm.*'].rows['.*(?<!_entry)$'].rows['.*(?<!_exit)$']
line.steering_monitors_x = tt_monitors.name
line.steering_monitors_y = tt_monitors.name

# Define elements to be used as correctors for orbit correction
# (for LHC all element namesstarting by "mcb.", containing "h." or "v.")
tt_h_correctors = tt.rows['mcb.*'].rows['.*h\..*']
line.steering_correctors_x = tt_h_correctors.name
tt_v_correctors = tt.rows['mcb.*'].rows['.*v\..*']
line.steering_correctors_y = tt_v_correctors.name

# Reference twiss (no misalignments)
tw_ref = line.twiss4d()

# Introduce misalignments on all quadrupoles
tt = line.get_table()
tt_quad = tt.rows[tt.element_type == 'Quadrupole']
shift_x = np.random.randn(len(tt_quad)) * 1e-3 # 1. mm rms shift on all quads
shift_y = np.random.randn(len(tt_quad)) * 1e-3 # 1. mm rms shift on all quads
for nn_quad, sx, sy in zip(tt_quad.name, shift_x, shift_y):
    line.element_refs[nn_quad].shift_x = sx
    line.element_refs[nn_quad].shift_y = sy

# Create orbit correction object without running the correction
orbit_correction = line.correct_trajectory(twiss_table=tw_ref, run=False)

# Inspect singular values of the response matrices
x_sv = orbit_correction.x_correction.singular_values
y_sv = orbit_correction.y_correction.singular_values

# Closed twiss fails (closed orbit is not found)
# line.twiss4d()

# Correct
orbit_correction.correct(n_singular_values=(200, 210))

# Remove applied correction
orbit_correction.clear_correction_knobs()

# Correct with a customized number of singular values
orbit_correction.correct(n_singular_values=(200, 210))

# Twiss after correction
tw_after = line.twiss4d()

# Extract correction strength
s_x_correctors = orbit_correction.x_correction.s_correctors
s_y_correctors = orbit_correction.y_correction.s_correctors
kicks_x = orbit_correction.x_correction.get_kick_values()
kicks_y = orbit_correction.y_correction.get_kick_values()

# Plots
import matplotlib.pyplot as plt
plt.close('all')

plt.figure(1, figsize=(6.4, 4.8*1.7))
sp1 = plt.subplot(411)
sp1.plot(tw_before.s, tw_before.x * 1e3, label='before corr.')
sp1.plot(tw_after.s, tw_after.x * 1e3, label='after corr.')
plt.legend(loc='upper right')
plt.ylabel('x [mm]')

sp2 = plt.subplot(412, sharex=sp1)
sp2.stem(s_x_correctors, kicks_x * 1e6)
plt.ylabel(r'x kick [$\mu$rad]')

sp3 = plt.subplot(413, sharex=sp1)
sp3.plot(tw_before.s, tw_before.y * 1e3)
sp3.plot(tw_after.s, tw_after.y * 1e3)
plt.ylabel('y [mm]')

sp4 = plt.subplot(414, sharex=sp1)
sp4.stem(s_y_correctors, kicks_y * 1e6)
plt.ylabel(r'y kick [$\mu$rad]')
sp4.set_xlabel('s [m]')

plt.subplots_adjust(hspace=0.3, top=0.95, bottom=0.08)

plt.figure(2)
plt.semilogy(np.abs(x_sv), '.-', label='x')
plt.semilogy(np.abs(y_sv), '.-', label='y')
plt.legend()
plt.xlabel('mode')
plt.ylabel('singular value (modulus)')

plt.show()