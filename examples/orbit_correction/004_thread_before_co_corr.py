import xtrack as xt
import numpy as np
from numpy.matlib import repmat

import orbit_correction as oc

line_range = ('ip2', 'ip3')
betx_start_guess = 1.
bety_start_guess = 1.

line_range = (None, None)
betx_start_guess = None
bety_start_guess = None

line = xt.Line.from_json(
    '../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')
tt = line.get_table().rows[line_range[0]:line_range[1]]
line.twiss_default['co_search_at'] = 'ip7'

tw = line.twiss4d(start=line_range[0], end=line_range[1],
                  betx=betx_start_guess,
                  bety=bety_start_guess)

# Select monitors by names (starting by "bpm" and not ending by "_entry" or "_exit")
tt_monitors = tt.rows['bpm.*'].rows['.*(?<!_entry)$'].rows['.*(?<!_exit)$']
monitor_names = tt_monitors.name

# Select h correctors by names (starting by "mcb.", containing "h.", and ending by ".b1")
# tt_h_correctors = tt.rows['mcb.*'].rows['.*h\..*'].rows['.*\.b1']
tt_h_correctors = tt.rows[tt.element_type == 'Quadrupole']
h_corrector_names = tt_h_correctors.name

# Select v correctors by names (starting by "mcb.", containing "v.", and ending by ".b1")
# tt_v_correctors = tt.rows['mcb.*'].rows['.*v\..*'].rows['.*\.b1']
tt_v_correctors = tt.rows[tt.element_type == 'Quadrupole']
v_corrector_names = tt_v_correctors.name

orbit_correction_h = oc.OrbitCorrection(line=line, plane='x', monitor_names=monitor_names,
                                        corrector_names=h_corrector_names,
                                        start=line_range[0], end=line_range[1])

orbit_correction_v = oc.OrbitCorrection(line=line, plane='y', monitor_names=monitor_names,
                                        corrector_names=v_corrector_names,
                                        start=line_range[0], end=line_range[1])

# Introduce some orbit perturbation

# h_kicks = {'mcbh.14r2.b1': 1e-5, 'mcbh.26l3.b1':-3e-5}
# v_kicks = {'mcbv.11r2.b1': -2e-5, 'mcbv.29l3.b1':-4e-5}

# for nn_kick, kick in h_kicks.items():
#     line.element_refs[nn_kick].knl[0] -= kick

# for nn_kick, kick in v_kicks.items():
#     line.element_refs[nn_kick].ksl[0] += kick

tt = line.get_table()
# tt_quad = tt.rows[tt.element_type == 'Quadrupole']
tt_quad = tt.rows['mq\..*']
shift_x = np.random.randn(len(tt_quad)) * 1e-3 # 1 mm rms shift on all quads
shift_y = np.random.randn(len(tt_quad)) * 1e-3 # 1 mm rms shift on all quads
for nn_quad, sx, sy in zip(tt_quad.name, shift_x, shift_y):
    line.element_refs[nn_quad].shift_x = sx
    line.element_refs[nn_quad].shift_y = sy



tt = line.get_table()
line_length = tt.s[-1]

ds_correction = 800
step_size = ds_correction

s_corr_end = ds_correction
s_corr_start = 0

i_win = 0
end_loop = False
while not end_loop:

    if s_corr_end > line_length:
        s_corr_end = line_length
        end_loop = True

    print(f'Window {i_win}, s_end: {s_corr_end}')
    tt_part = tt.rows[s_corr_start:s_corr_end:'s']
    start = tt_part.name[0]
    end = tt_part.name[-1]
    these_h_corrector_names = [name for name in h_corrector_names if
                               name in tt_part.name]
    these_v_corrector_names = [name for name in v_corrector_names if
                                 name in tt_part.name]
    these_monitor_names = [name for name in monitor_names if name in tt_part.name]

    tt_new_part = tt.rows[s_corr_end-ds_correction:s_corr_end:'s']
    start_new = tt_new_part.name[0]
    end_new = tt_new_part.name[-1]
    these_h_corrector_names_new = [name for name in h_corrector_names if
                                   name in tt_new_part.name]
    these_v_corrector_names_new = [name for name in v_corrector_names if
                                      name in tt_new_part.name]
    these_monitor_names_new = [name for name in monitor_names if name in tt_new_part.name]

    try:

        this_ocorr_h_new = oc.OrbitCorrection(
            line=line, plane='x', monitor_names=these_monitor_names_new,
            corrector_names=these_h_corrector_names_new,
            start=start_new, end=end_new)

        this_ocorr_v_new = oc.OrbitCorrection(
            line=line, plane='y', monitor_names=these_monitor_names_new,
            corrector_names=these_v_corrector_names_new,
            start=start_new, end=end_new)

        this_ocorr_h_new.correct(rcond=1e-1)
        this_ocorr_v_new.correct(rcond=1e-1)

        this_ocorr_h = oc.OrbitCorrection(
            line=line, plane='x', monitor_names=these_monitor_names,
            corrector_names=these_h_corrector_names,
            start=start, end=end)

        this_ocorr_v = oc.OrbitCorrection(
            line=line, plane='y', monitor_names=these_monitor_names,
            corrector_names=these_v_corrector_names,
            start=start, end=end)

        this_ocorr_h.correct(rcond=1e-1)
        this_ocorr_v.correct(rcond=1e-1)


        s_corr_end += ds_correction
        step_size = ds_correction
        i_win += 1
    except NotImplementedError:
        step_size /= 2
        s_corr_end -= step_size

two = line.twiss(only_orbit=True, start=line.element_names[0],
                 end=line.element_names[-1], betx=1, bety=1,
                 _continue_if_lost=True)

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
sp1 = plt.subplot(311)
plt.plot(two.s, two.x, label='x')
plt.plot(two.s, two.y, label='y')
sp2 = plt.subplot(312, sharex=sp1)
plt.stem(this_ocorr_h.s_correctors, this_ocorr_v.get_kick_values())
sp3 = plt.subplot(313, sharex=sp1)
plt.stem(this_ocorr_v.s_correctors, this_ocorr_v.get_kick_values())

plt.show()

prrrr


tw_meas = line.twiss4d(only_orbit=True, start=line_range[0], end=line_range[1],
                          betx=betx_start_guess,
                          bety=bety_start_guess)

x_meas = tw_meas.rows[monitor_names].x
y_meas = tw_meas.rows[monitor_names].y
s_meas = tw_meas.rows[monitor_names].s

n_micado = None

for iter in range(10):
    orbit_correction_h.correct()
    orbit_correction_v.correct()

    tw_after = line.twiss4d(only_orbit=True, start=line_range[0], end=line_range[1],
                            betx=betx_start_guess,
                            bety=bety_start_guess)
    print('max x: ', tw_after.x.max())
    print('max y: ', tw_after.y.max())

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