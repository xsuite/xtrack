import xtrack as xt
import numpy as np

line = xt.load(
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
rgen = np.random.RandomState(1) # fix seed for random number generator
shift_x = rgen.randn(len(tt_quad)) * 0.01e-3 # 0.01 mm rms shift on all quads
shift_y = rgen.randn(len(tt_quad)) * 0.01e-3 # 0.01 mm rms shift on all quads
for nn_quad, sx, sy in zip(tt_quad.name, shift_x, shift_y):
    line[nn_quad].shift_x = sx
    line[nn_quad].shift_y = sy

# Twiss before correction
tw_before = line.twiss4d()

# Correct orbit using 5 correctors in each plane
orbit_correction = line.correct_trajectory(twiss_table=tw_ref, n_micado=5,
                                           n_iter=1)
# prints:
#
# Iteration 0, x_rms: 1.65e-03 -> 1.06e-04, y_rms: 2.25e-03 -> 1.76e-04

# Twiss after correction
tw_after = line.twiss4d()

# Extract correction strength
s_x_correctors = orbit_correction.x_correction.s_correctors
s_y_correctors = orbit_correction.y_correction.s_correctors
kicks_x = orbit_correction.x_correction.get_kick_values()
kicks_y = orbit_correction.y_correction.get_kick_values()

#!end-doc-part

# Plots
import matplotlib.pyplot as plt
plt.close('all')

plt.figure(1, figsize=(6.4, 4.8*1.7))
sp1 = plt.subplot(411)
sp1.plot(tw_before.s, tw_before.x * 1e3,
        label=f'before corr. (rms: {tw_before.x.std() * 1e3:.2e} mm)')
sp1.plot(tw_after.s, tw_after.x * 1e3,
        label=f'after corr. (rms: {tw_after.x.std() * 1e3:.2e} mm)')
plt.legend(loc='lower right')
plt.ylabel('x [mm]')

sp2 = plt.subplot(412, sharex=sp1)
sp2.stem(s_x_correctors, kicks_x * 1e6)
plt.ylabel(r'x kick [$\mu$rad]')

sp3 = plt.subplot(413, sharex=sp1)
sp3.plot(tw_before.s, tw_before.y * 1e3,
        label=f'before corr. (rms: {tw_before.y.std() * 1e3:.2e} mm)')
sp3.plot(tw_after.s, tw_after.y * 1e3,
        label=f'after corr. (rms: {tw_after.y.std() * 1e3:.2e} mm)')
plt.ylabel('y [mm]')
plt.legend(loc='lower right')

sp4 = plt.subplot(414, sharex=sp1)
sp4.stem(s_y_correctors, kicks_y * 1e6)
plt.ylabel(r'y kick [$\mu$rad]')
sp4.set_xlabel('s [m]')

plt.subplots_adjust(hspace=0.3, top=0.95, bottom=0.08)

plt.show()