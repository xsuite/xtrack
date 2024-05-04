import xtrack as xt
import numpy as np
from numpy.matlib import repmat

import orbit_correction as oc

line = xt.Line.from_json(
    '../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')
tt = line.get_table()
line.twiss_default['co_search_at'] = 'ip7'

tw = line.twiss4d()

# Select monitors by names (starting by "bpm" and not ending by "_entry" or "_exit")
tt_monitors = tt.rows['bpm.*'].rows['.*(?<!_entry)$'].rows['.*(?<!_exit)$']
h_monitor_names = tt_monitors.name

# Select h correctors by names (starting by "mcb.", containing "h.", and ending by ".b1")
tt_h_correctors = tt.rows['mcb.*'].rows['.*h\..*'].rows['.*\.b1']
h_corrector_names = tt_h_correctors.name

orbit_correction = oc.OrbitCorrection(line=line, h_monitor_names=h_monitor_names,
                                        h_corrector_names=h_corrector_names)
orbit_correction.add_correction_knobs()

response_matrix_x = oc._build_response_matrix(
    tw, h_monitor_names, h_corrector_names, mode='closed')

# Introduce some orbit perturbation

h_kicks = {'mcbh.14r2.b1': 1e-5, 'mcbh.26l3.b1':-3e-5}
kick_vect_x = np.zeros(response_matrix_x.shape[1])

for nn_kick, kick in h_kicks.items():
    line.element_refs[nn_kick].knl[0] -= kick
    i_h_kick = np.where(h_corrector_names == nn_kick)[0][0]
    kick_vect_x[i_h_kick] = kick

# tt = line.get_table()
# tt_quad = tt.rows[tt.element_type == 'Quadrupole']
# shift_x = np.random.randn(len(tt_quad)) * 1e-5 # 10 um rm shift on all quads
# for nn_quad, shift in zip(tt_quad.name, shift_x):
#     line.element_refs[nn_quad].shift_x = shift

tw_meas = line.twiss4d(only_orbit=True)

x_meas = tw_meas.rows[h_monitor_names].x
s_x_meas = tw_meas.rows[h_monitor_names].s

n_micado = None

for iter in range(3):
    # Measure the orbit
    tw_iter = line.twiss4d(only_orbit=True)

    x_iter = tw_iter.rows[h_monitor_names].x

    correction_x = oc._compute_correction(x_iter, response_matrix_x, n_micado)

    # Apply correction
    orbit_correction.apply_correction(correction_x)

    tw_after = line.twiss4d(only_orbit=True)

    print('max x: ', tw_after.x.max())

x_meas_after = tw_after.rows[h_monitor_names].x

s_correctors = tw_after.rows[h_corrector_names].s

# Extract kicks from the knobs
applied_kicks = orbit_correction.get_kick_values()

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
sp1 = plt.subplot(211)
sp1.plot(s_x_meas, x_meas, label='measured')
sp1.plot(s_x_meas, x_meas_after, label='corrected')

sp2 = plt.subplot(212, sharex=sp1)
sp2.plot(s_correctors, applied_kicks, label='applied kicks')

plt.show()