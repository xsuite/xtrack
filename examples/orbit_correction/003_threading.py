import xtrack as xt
import numpy as np
from numpy.matlib import repmat

import orbit_correction as oc

line_range = ('ip2', 'ip3')
betx_start_guess = 1.
bety_start_guess = 1.

line = xt.Line.from_json(
    '../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')
tt = line.get_table().rows[line_range[0]:line_range[1]]
line.twiss_default['co_search_at'] = 'ip7'

tw = line.twiss4d(start=line_range[0], end=line_range[1],
                  betx=betx_start_guess,
                  bety=bety_start_guess)

# Select monitors by names (starting by "bpm" and not ending by "_entry" or "_exit")
tt_monitors = tt.rows['bpm.*'].rows['.*(?<!_entry)$'].rows['.*(?<!_exit)$']
h_monitor_names = tt_monitors.name

# Select h correctors by names (starting by "mcb.", containing "h.", and ending by ".b1")
tt_h_correctors = tt.rows['mcb.*'].rows['.*h\..*'].rows['.*\.b1']
h_corrector_names = tt_h_correctors.name

# Add correction knobs to the correctors
h_correction_knobs = []
for nn_kick in h_corrector_names:
    corr_knob_name = f'orbit_corr_{nn_kick}'
    assert hasattr(line[nn_kick], 'knl')
    line.vars[corr_knob_name] = 0
    line.element_refs[nn_kick].knl[0] += line.vars[f'orbit_corr_{nn_kick}']
    h_correction_knobs.append(corr_knob_name)


# Wille eq. 3.164
response_matrix_x = oc._build_response_matrix(
    tw, h_monitor_names, h_corrector_names, mode='open')

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

tw_meas = line.twiss4d(only_orbit=True, start=line_range[0], end=line_range[1],
                          betx=betx_start_guess,
                          bety=bety_start_guess)

x_meas = tw_meas.rows[h_monitor_names].x
s_x_meas = tw_meas.rows[h_monitor_names].s

n_micado = None

for iter in range(3):
    # Measure the orbit
    tw_iter = line.twiss4d(only_orbit=True,
                            start=line_range[0], end=line_range[1],
                            betx=betx_start_guess,
                            bety=bety_start_guess)

    x_iter = tw_iter.rows[h_monitor_names].x

    correction_x = oc._compute_correction(x_iter, response_matrix_x, n_micado)

    # correction_masked, residual_x, rank_x, sval_x = np.linalg.lstsq(
    #             response_matrix_x[2:, :], -x_iter[2:], rcond=None)
    # correction_x = np.zeros(n_hcorrectors)
    # correction_x = correction_masked

    # Apply correction
    for nn_knob, kick in zip(h_correction_knobs, correction_x):
        line.vars[nn_knob] -= kick # knl[0] is -kick

    tw_after = line.twiss4d(only_orbit=True, start=line_range[0], end=line_range[1],
                            betx=betx_start_guess,
                            bety=bety_start_guess)

    print('max x: ', tw_after.x.max())

x_meas_after = tw_after.rows[h_monitor_names].x

s_correctors = tw_after.rows[h_corrector_names].s

# Extract kicks from the knobs
applied_kicks = np.array([line.vv[nn_knob] for nn_knob in h_correction_knobs])

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
sp1 = plt.subplot(211)
sp1.plot(s_x_meas, x_meas, label='measured')
sp1.plot(s_x_meas, x_meas_after, label='corrected')

sp2 = plt.subplot(212, sharex=sp1)
sp2.plot(s_correctors, applied_kicks, label='applied kicks')

plt.show()