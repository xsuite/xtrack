import numpy as np
from cpymad.madx import Madx
import xtrack as xt

mad_ti2 = Madx()
mad_ti2.call('../../test_data/sps_to_lhc_ti2/ti2.seq')
mad_ti2.call('../../test_data/sps_to_lhc_ti2/ti2_liu.str')
mad_ti2.beam()
mad_ti2.use('ti2')

line = xt.Line.from_madx_sequence(mad_ti2.sequence['ti2'])
line.particle_ref = xt.Particles(p0c=450e9, mass0=xt.PROTON_MASS_EV, q0=1)
tt = line.get_table()

# Define elements to be used as monitors for orbit correction
# (in this case all element names starting by "bpm" and not ending by "_entry" or "_exit")
tt_monitors = tt.rows['bpm.*'].rows['.*(?<!_entry)$'].rows['.*(?<!_exit)$']
line.steering_monitors_x = tt_monitors.name
line.steering_monitors_y = tt_monitors.name

# Define elements to be used as correctors for orbit correction
# (in this case all element names starting by "mci.", containing "h." or "v.")
tt_h_correctors = tt.rows['mci.*'].rows['.*h\..*']
line.steering_correctors_x = tt_h_correctors.name
tt_v_correctors = tt.rows['mci.*'].rows['.*v\..*']
line.steering_correctors_y = tt_v_correctors.name

# Initial conditions from upstream ring
init = xt.TwissInit(betx=27.77906807, bety=120.39920690,
                     alfx=0.63611880, alfy=-2.70621900,
                     dx=-0.59866300, dpx=0.01603536)

# Reference twiss (no misalignments)
tw_ref = line.twiss4d(start='ti2$start', end='ti2$end', init=init)

# Introduce misalignments on all quadrupoles
tt = line.get_table()
tt_quad = tt.rows['mqi.*']
rgen = np.random.RandomState(2) # fix seed for random number generator
shift_x = rgen.randn(len(tt_quad)) * 0.1e-3 # 0.1 mm rms shift on all quads
shift_y = rgen.randn(len(tt_quad)) * 0.1e-3 # 0.1 mm rms shift on all quads
for nn_quad, sx, sy in zip(tt_quad.name, shift_x, shift_y):
    line.element_refs[nn_quad].shift_x = sx
    line.element_refs[nn_quad].shift_y = sy

# Twiss before correction
tw_before = line.twiss4d(start='ti2$start', end='ti2$end', init=init)

# Correct trajectory
correction = line.correct_trajectory(twiss_table=tw_ref, start='ti2$start', end='ti2$end')
# prints:
#
# Iteration 0, x_rms: 2.01e-03 -> 1.80e-04, y_rms: 6.94e-04 -> 1.23e-04
# Iteration 1, x_rms: 1.80e-04 -> 1.80e-04, y_rms: 1.23e-04 -> 1.23e-04

# Twiss after correction
tw_after = line.twiss4d(start='ti2$start', end='ti2$end', init=init)

# Extract correction strength
s_x_correctors = correction.x_correction.s_correctors
s_y_correctors = correction.y_correction.s_correctors
kicks_x = correction.x_correction.get_kick_values()
kicks_y = correction.y_correction.get_kick_values()

#!end-doc-part

# Plots
import matplotlib.pyplot as plt
plt.close('all')

plt.figure(1, figsize=(6.4, 4.8*1.7))
sp1 = plt.subplot(411)
sp1.plot(tw_before.s, tw_before.x * 1e3, label='before corr.')
sp1.plot(tw_after.s, tw_after.x * 1e3, label='after corr.')
plt.legend(loc='upper left')
plt.ylabel('x [mm]')

sp2 = plt.subplot(412, sharex=sp1)
sp2.stem(s_x_correctors, kicks_x * 1e6)
plt.ylabel(r'x kick [$\mu$rad]')

sp3 = plt.subplot(413, sharex=sp1, sharey=sp1)
sp3.plot(tw_before.s, tw_before.y * 1e3)
sp3.plot(tw_after.s, tw_after.y * 1e3)
plt.ylabel('y [mm]')

sp4 = plt.subplot(414, sharex=sp1)
sp4.stem(s_y_correctors, kicks_y * 1e6)
plt.ylabel(r'y kick [$\mu$rad]')
sp4.set_xlabel('s [m]')

plt.subplots_adjust(hspace=0.3, top=0.95, bottom=0.08)

plt.show()
