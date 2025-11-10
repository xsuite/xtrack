import xtrack as xt
import numpy as np

line = xt.load('../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')
tt = line.get_table()

# Define elements to be used as monitors for orbit correction
# (for LHC all element names starting by "bpm" and not ending by "_entry" or "_exit")
tt_monitors = tt.rows['bpm.*'].rows['.*(?<!_entry)$'].rows['.*(?<!_exit)$']
line.steering_monitors_x = tt_monitors.name
line.steering_monitors_y = tt_monitors.name

# Define elements to be used as correctors for orbit correction
# (for LHC all element names starting by "mcb.", containing "h." or "v.")
tt_h_correctors = tt.rows['mcb.*'].rows['.*h\..*']
line.steering_correctors_x = tt_h_correctors.name
tt_v_correctors = tt.rows['mcb.*'].rows['.*v\..*']
line.steering_correctors_y = tt_v_correctors.name

# Reference twiss (no misalignments)
tw_ref = line.twiss4d()

# Create orbit correction object without running the correction
orbit_correction = line.correct_trajectory(twiss_table=tw_ref, run=False,
                                           n_singular_values=250)

# Inspect singular values of the response matrices
x_sv = orbit_correction.x_correction.singular_values
y_sv = orbit_correction.y_correction.singular_values

delta0 = 1e-4

# Twiss before correction
tw_before = line.twiss4d(delta0=delta0)

# Correct
orbit_correction.correct(delta0=delta0)

# Twiss after correction
tw_after = line.twiss4d(delta0=delta0)

# Extract correction strength
s_x_correctors = orbit_correction.x_correction.s_correctors
s_y_correctors = orbit_correction.y_correction.s_correctors
kicks_x = orbit_correction.x_correction.get_kick_values()
kicks_y = orbit_correction.y_correction.get_kick_values()

#!end-doc-part

# Plots
import matplotlib.pyplot as plt
plt.close('all')

fig1 = plt.figure(1, figsize=(6.4, 4.8*1.7))
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

fig2 = plt.figure(2)
plt.semilogy(np.abs(x_sv), '.-', label='x')
plt.semilogy(np.abs(y_sv), '.-', label='y')
plt.legend()
plt.xlabel('mode')
plt.ylabel('singular value (modulus)')

plt.show()