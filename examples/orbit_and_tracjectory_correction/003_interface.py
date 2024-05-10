import xtrack as xt
import numpy as np
from numpy.matlib import repmat

import orbit_correction as oc

line_range = ('ip2', 'ip3')
betx_start_guess = 1.
bety_start_guess = 1.

# line_range = (None, None)
# betx_start_guess = None
# bety_start_guess = None

# line_range = ('ip6', 'ip8')
# betx_start_guess = 1.
# bety_start_guess = 1.

line = xt.Line.from_json(
    '../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')


tt = line.get_table().rows[line_range[0]:line_range[1]]
# I use the twiss to allow the loop around of the ring
# tt = line.twiss4d(start=line_range[0], end=line_range[1],
#                     betx=betx_start_guess,
#                     bety=bety_start_guess)

line.twiss_default['co_search_at'] = 'ip7'

tw = line.twiss4d(start=line_range[0], end=line_range[1],
                  betx=betx_start_guess,
                  bety=bety_start_guess)

# Select monitors by names (starting by "bpm" and not ending by "_entry" or "_exit")
tt_monitors = tt.rows['bpm.*'].rows['.*(?<!_entry)$'].rows['.*(?<!_exit)$']
monitor_names = tt_monitors.name

# Select h correctors by names (starting by "mcb.", containing "h.", and ending by ".b1")
tt_h_correctors = tt.rows['mcb.*'].rows['.*h\..*']
# tt_h_correctors = tt.rows[tt.element_type == 'Quadrupole']
h_corrector_names = tt_h_correctors.name

# Select v correctors by names (starting by "mcb.", containing "v.", and ending by ".b1")
tt_v_correctors = tt.rows['mcb.*'].rows['.*v\..*']
# tt_v_correctors = tt.rows[tt.element_type == 'Quadrupole']
v_corrector_names = tt_v_correctors.name



# Introduce some orbit perturbation

# h_kicks = {'mcbh.14r2.b1': 1e-5, 'mcbh.26l3.b1':-3e-5}
# v_kicks = {'mcbv.11r2.b1': -2e-5, 'mcbv.29l3.b1':-4e-5}

# for nn_kick, kick in h_kicks.items():
#     line.element_refs[nn_kick].knl[0] -= kick

# for nn_kick, kick in v_kicks.items():
#     line.element_refs[nn_kick].ksl[0] += kick

tt = line.get_table()
tt_quad = tt.rows[tt.element_type == 'Quadrupole']
shift_x = np.random.randn(len(tt_quad)) * 1e-3 # 10 um rm shift on all quads
shift_y = np.random.randn(len(tt_quad)) * 0  # 10 um rm shift on all quads
for nn_quad, sx, sy in zip(tt_quad.name, shift_x, shift_y):
    line.element_refs[nn_quad].shift_x = sx
    line.element_refs[nn_quad].shift_y = sy

tw_meas = line.twiss4d(only_orbit=True, start=line_range[0], end=line_range[1],
                          betx=betx_start_guess,
                          bety=bety_start_guess)

x_meas = tw_meas.rows[monitor_names].x
y_meas = tw_meas.rows[monitor_names].y
s_meas = tw_meas.rows[monitor_names].s

n_micado = None

for iter in range(10):
    orbit_correction_h = oc.OrbitCorrectionSinglePlane(line=line, plane='x', monitor_names=monitor_names,
                                        corrector_names=h_corrector_names,
                                        start=line_range[0], end=line_range[1])

    orbit_correction_v = oc.OrbitCorrectionSinglePlane(line=line, plane='y', monitor_names=monitor_names,
                                            corrector_names=v_corrector_names,
                                            start=line_range[0], end=line_range[1])
    orbit_correction_h.correct()
    orbit_correction_v.correct()

    tw_after = line.twiss4d(only_orbit=True, start=line_range[0], end=line_range[1],
                            betx=betx_start_guess,
                            bety=bety_start_guess)
    print(f'max x: {tw_after.x.max()}    max y: {tw_after.y.max()}, rms x: {tw_after.x.std()}    rms y: {tw_after.y.std()}')


x_meas_after = tw_after.rows[monitor_names].x
y_meas_after = tw_after.rows[monitor_names].y

s_h_correctors = tw_after.rows[h_corrector_names].s
s_v_correctors = tw_after.rows[v_corrector_names].s

# Extract kicks from the knobs
applied_kicks_h = orbit_correction_h.get_kick_values()
applied_kicks_v = orbit_correction_v.get_kick_values()

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1, figsize=(6.4, 4.8*1.7))
sp1 = plt.subplot(411)
sp1.plot(s_meas, x_meas, label='measured')
sp1.plot(s_meas, x_meas_after, label='corrected')

sp2 = plt.subplot(412, sharex=sp1)
markerline, stemlines, baseline = sp2.stem(s_h_correctors, applied_kicks_h, label='applied kicks')

sp3 = plt.subplot(413, sharex=sp1)
sp3.plot(s_meas, y_meas, label='measured')
sp3.plot(s_meas, y_meas_after, label='corrected')

sp4 = plt.subplot(414, sharex=sp1)
markerline, stemlines, baseline = sp4.stem(s_v_correctors, applied_kicks_v, label='applied kicks')
plt.subplots_adjust(hspace=0.35, top=.90, bottom=.10)
plt.show()