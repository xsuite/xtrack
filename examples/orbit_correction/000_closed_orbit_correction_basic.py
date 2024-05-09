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
# tt_quad = tt.rows['mq\..*']
shift_x = np.random.randn(len(tt_quad)) * 0.1e-3 # 1 mm rms shift on all quads
shift_y = np.random.randn(len(tt_quad)) * 0.1e-3 # 1 mm rms shift on all quads
for nn_quad, sx, sy in zip(tt_quad.name, shift_x, shift_y):
    line.element_refs[nn_quad].shift_x = sx
    line.element_refs[nn_quad].shift_y = sy

# Twiss before matching
tw_before = line.twiss4d()

# Correct orbit
orbit_correction = line.correct_trajectory(twiss_table=tw_ref)


# Twiss after matching
tw_after = line.twiss4d()


prrrr



# Introduce some orbit perturbation

# h_kicks = {'mcbh.14r2.b1': 1e-5, 'mcbh.26l3.b1':-3e-5}
# v_kicks = {'mcbv.11r2.b1': -2e-5, 'mcbv.29l3.b1':-4e-5}

# for nn_kick, kick in h_kicks.items():
#     line.element_refs[nn_kick].knl[0] -= kick

# for nn_kick, kick in v_kicks.items():
#     line.element_refs[nn_kick].ksl[0] += kick




threader = orbit_correction.thread(ds_thread=500., rcond_short=None, rcond_long=None)

kick_h_after_thread = threader.x_correction.get_kick_values()
kick_v_after_thread = threader.y_correction.get_kick_values()

tw_meas = line.twiss4d(only_orbit=True)
x_meas = tw_meas.rows[monitor_names].x
y_meas = tw_meas.rows[monitor_names].y
s_meas = tw_meas.rows[monitor_names].s

n_micado = None

for iter in range(5):
    orbit_correction.correct()

    tw_after = line.twiss4d(only_orbit=True)
    print(f'max x: {tw_after.x.max()}    max y: {tw_after.y.max()}, rms x: {tw_after.x.std()}    rms y: {tw_after.y.std()}')

# orbit_correction_h._measure_position()
# x_bare = orbit_correction_h.position + orbit_correction_h.response_matrix @ orbit_correction_h.get_kick_values()
# orbit_correction_h._compute_correction(position=x_bare)
# orbit_correction_h._clean_correction_knobs()
# orbit_correction_h._apply_correction()

tw_after = line.twiss4d(only_orbit=True)

x_meas_after = tw_after.rows[monitor_names].x
y_meas_after = tw_after.rows[monitor_names].y

s_h_correctors = tw_after.rows[h_corrector_names].s
s_v_correctors = tw_after.rows[v_corrector_names].s

# Extract kicks from the knobs
applied_kicks_h = orbit_correction.x_correction.get_kick_values()
applied_kicks_v = orbit_correction.y_correction.get_kick_values()


import matplotlib.pyplot as plt
plt.close('all')


plt.figure(2, figsize=(6.4, 4.8*1.7))
sp1 = plt.subplot(411)
sp1.plot(s_meas, x_meas, label='measured')
sp1.plot(s_meas, x_meas_after, label='corrected')


sp2 = plt.subplot(412, sharex=sp1)
markerline, stemlines, baseline = sp2.stem(s_h_correctors, applied_kicks_h, label='applied kicks')
plt.plot(orbit_correction.x_correction.s_correctors, kick_h_after_thread, 'xr')

sp3 = plt.subplot(413, sharex=sp1)
sp3.plot(s_meas, y_meas, label='measured')
sp3.plot(s_meas, y_meas_after, label='corrected')

sp4 = plt.subplot(414, sharex=sp1)
markerline, stemlines, baseline = sp4.stem(s_v_correctors, applied_kicks_v, label='applied kicks')
plt.plot(orbit_correction.y_correction.s_correctors, kick_v_after_thread, 'xr')
plt.subplots_adjust(hspace=0.35, top=.90, bottom=.10)
plt.show()