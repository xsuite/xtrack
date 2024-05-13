import xtrack as xt
import xobjects as xo
import numpy as np

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
rgen = np.random.RandomState(1) # fix seed for random number generator
shift_x = rgen.randn(len(tt_quad)) * 0.01e-3 # 0.01 mm rms shift on all quads
shift_y = rgen.randn(len(tt_quad)) * 0.01e-3 # 0.01 mm rms shift on all quads
for nn_quad, sx, sy in zip(tt_quad.name, shift_x, shift_y):
    line.element_refs[nn_quad].shift_x = sx
    line.element_refs[nn_quad].shift_y = sy

# Create orbit correction object without running the correction
orbit_correction = line.correct_trajectory(twiss_table=tw_ref, run=False)

# Inspect singular values of the response matrices
x_sv = orbit_correction.x_correction.singular_values
y_sv = orbit_correction.y_correction.singular_values

# Twiss before correction
tw_before = line.twiss4d()

# Correct
orbit_correction.correct()

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

assert tw_before.x.std() > 1e-3
assert tw_before.y.std() > 1e-3
assert tw_after.y.std() < 10e-6
assert tw_after.x.std() < 10e-6

assert kicks_x.std() < 5e-7
assert kicks_y.std() < 5e-7

xo.assert_allclose(np.abs(kicks_x).max(), 2e-6, atol=2e-6)
xo.assert_allclose(np.abs(kicks_y).max(), 2e-6, atol=1e-6)