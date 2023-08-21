import numpy as np
import xtrack as xt

line = xt.Line.from_json('fccee_p_ring_thin_wig.json')
line.build_tracker()
varvals = line._xdeps_vref._owner.copy()

# Add longer wigglers
section_list = [
    ('qrfr1.1_entry', 'qrfl1.1_exit'),
    # ('qrfr1.2_entry', 'qrfl1.2_exit'), # taken by the RF
    ('qrfr1.3_entry', 'qrfl1.3_exit'),
    ('qrfr1.4_entry', 'qrfl1.4_exit'),
]


n_slices = 10
l_add = 0
line.vars['k0l_long_wig'] = 0
line.vars['wig_drift_fraction'] = 0.7
line.vars['wig_core_len_factor'] = 1
line.vars['wig_edge_len_factor'] = 1
for iss, section in enumerate(section_list):

    tt = line.get_table()
    tt_insertion = tt.rows[section[0]: section[1]]
    mask_drift_q = tt_insertion.mask['drift_q.*']
    tt_wig_drifts = tt_insertion.rows[~mask_drift_q].rows['drift_.*']

    for jj, nn in enumerate(tt_wig_drifts.name):
        s_start_wig = tt['s', nn]
        s_end_wig = tt['s', nn] + line[nn].length

        tag_wig = f'w{iss}_{jj}'

        n_kicks = 4
        s_wig_kicks = np.linspace(s_start_wig - l_add, s_end_wig + l_add, n_kicks)
        s_wig_plus = s_wig_kicks[:-1:2]
        s_wig_minus = s_wig_kicks[1::2]

        line.discard_tracker()

        kk_wig = f'k0l_wig.{tag_wig}'
        line.vars[kk_wig] = line.vars['k0l_long_wig']
        for ii, (ss_p, ss_m) in enumerate(zip(s_wig_plus, s_wig_minus)):
            half_period = ss_m - ss_p
            if ii == 0:
                ss_p += 0.5 * half_period
            if ii == len(s_wig_plus) - 1:
                ss_m -= 0.5 * half_period

            for i_slice in range(n_slices):
                print(
                    f'Wig {tag_wig} period {ii}/{len(s_wig_plus)} slice {i_slice}/{n_slices}      ',
                    end='\r', flush=True)
                wmg_plus = xt.Multipole(knl=[0])
                wmg_minus = xt.Multipole(knl=[0])

                nn_plus = f'mwg.{tag_wig}.plus.{ii}..{i_slice}'
                nn_minus = f'mwg.{tag_wig}.minus.{ii}..{i_slice}'
                line.insert_element(nn_plus, wmg_plus,
                    at_s=ss_p + i_slice * (1 - line.varval['wig_drift_fraction']) * half_period / n_slices)
                line.insert_element(nn_minus, wmg_minus,
                    at_s=ss_m + i_slice * (1 - line.varval['wig_drift_fraction']) * half_period / n_slices)
                line.element_refs[nn_plus].ksl[0] = line.vars[kk_wig] / n_slices
                line.element_refs[nn_minus].ksl[0] = -line.vars[kk_wig] / n_slices
                line.element_refs[nn_plus].length = half_period * (1 - line.vars['wig_drift_fraction']) / n_slices * line.vars['wig_core_len_factor']
                line.element_refs[nn_minus].length = half_period * (1 - line.vars['wig_drift_fraction']) / n_slices * line.vars['wig_core_len_factor']
                if ii == 0:
                    line.element_refs[nn_plus].ksl[0] = line.vars[kk_wig] / n_slices * 0.5
                    line.element_refs[nn_plus].length = half_period / n_slices * line.vars['wig_edge_len_factor'] # To make them weaker (as thy have low dispersion)
                if ii == len(s_wig_plus) - 1:
                    line.element_refs[nn_minus].ksl[0] = -line.vars[kk_wig] / n_slices * 0.5
                    line.element_refs[nn_minus].length = half_period / n_slices * line.vars['wig_edge_len_factor']# To make them weaker (as thy have low dispersion)

line.to_json('fccee_p_ring_thin_wig_long.json')

prrrr

line = line.from_json('fccee_p_ring_thin_wig_long.json')

line.vars['k0l_long_wig'] = 12e-4
line.vars['wig_core_len_factor'] = 1
line.vars['wig_edge_len_factor'] = 1

tt = line.get_table()
for nn in tt.rows['mwg.*'].name:
    line.element_refs[nn].hyl = line.element_refs[nn].ksl[0]._expr

line.build_tracker() # Important to do it after setting l_rad to update cached values

tw = line.twiss(method='4d')

line.configure_radiation(model='mean')
line.compensate_radiation_energy_loss()

tw_rad = line.twiss(eneloss_and_damping=True)
ex = tw_rad.nemitt_x_rad / (tw_rad.gamma0 * tw_rad.beta0)
ey = tw_rad.nemitt_y_rad / (tw_rad.gamma0 * tw_rad.beta0)
ez = tw_rad.nemitt_zeta_rad / (tw_rad.gamma0 * tw_rad.beta0)

print(f'ex = {ex:.2e} m, ey = {ey:.2e} m, ez = {ez:.2e} m')
print('damping times: ',
        tw_rad.damping_constants_s[0],
        tw_rad.damping_constants_s[1],
        tw_rad.damping_constants_s[2])

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1, figsize=(6.4, 4.8*1.5))
sp1 = plt.subplot(4,1,1)
sp1.plot(tw.s, tw_rad.betx, label='betx')
sp1.plot(tw.s, tw_rad.bety, label='bety')

sp2 = plt.subplot(4,1,2, sharex=sp1)
sp2.plot(tw.s, tw_rad.dx, label='dx')

sp3 = plt.subplot(4,1,3, sharex=sp1)
sp3.plot(tw.s, tw_rad.dy, label='dy')

sp4 = plt.subplot(4,1,4, sharex=sp1)
sp4.plot(tw.s, tw_rad.delta, label='delta')
plt.subplots_adjust(hspace=0.33, bottom=0.07)

plt.figure(2)
plt.subplot(3,1,1, sharex=sp1)
plt.plot(tw.s, tw_rad.betx, label='betx')
plt.plot(tw.s, tw_rad.bety, label='bety')

plt.subplot(3,1,2, sharex=sp1)
plt.plot(tw.s, tw_rad.betx/tw.betx - 1, label='betx')

plt.subplot(3,1,3, sharex=sp1)
plt.plot(tw.s, tw_rad.bety/tw.bety - 1, label='bety')

plt.show()


pppppp
# Tracking

line.configure_radiation(model='quantum')
p = line.build_particles(num_particles=30)
line.track(p, num_turns=400, turn_by_turn_monitor=True, time=True)
print(f'Tracking time: {line.time_last_track}')


dl = line.tracker._tracker_data_base.cache['dl_radiation']
hxl = line.tracker._tracker_data_base.cache['hxl_radiation']
hyl = line.tracker._tracker_data_base.cache['hyl_radiation']
px_co = tw_rad.px
py_co = tw_rad.py

mask = (dl != 0)
hx = np.zeros(shape=(len(dl),), dtype=np.float64)
hy = np.zeros(shape=(len(dl),), dtype=np.float64)
hx[mask] = (np.diff(px_co)[mask] + hxl[mask]) / dl[mask]
hy[mask] = (np.diff(py_co)[mask] + hyl[mask]) / dl[mask]
hh = np.sqrt(hx**2 + hy**2)

twe = tw_rad.rows[:-1]
cur_H_x = twe.gamx * twe.dx**2 + 2 * twe.alfx * twe.dx * twe.dpx + twe.betx * twe.dpx**2
I5_x  = np.sum(cur_H_x * hh**3 * dl)
I2_x = np.sum(hh**2 * dl)
I4_x = np.sum(twe.dx * hh**3 * dl) # to be generalized for combined function magnets

cur_H_y = twe.gamy * twe.dy**2 + 2 * twe.alfy * twe.dy * twe.dpy + twe.bety * twe.dpy**2
I5_y  = np.sum(cur_H_y * hh**3 * dl)
I2_y = np.sum(hh**2 * dl)
I4_y = np.sum(twe.dy * hh**3 * dl) # to be generalized for combined function magnets

lam_comp = 2.436e-12 # [m]
gamma0 = tw_rad.gamma0
ex_hof = 55 * np.sqrt(3) / 96 * lam_comp / 2 / np.pi * gamma0**2 * I5_x / (I2_x - I4_x)
ey_hof = 55 * np.sqrt(3) / 96 * lam_comp / 2 / np.pi * gamma0**2 * I5_y / (I2_y - I4_y)

mon = line.record_last_track

import matplotlib.pyplot as plt
plt.close('all')
fig = plt.figure(1)
spx = fig. add_subplot(3, 1, 1)
spx.plot(np.std(mon.x, axis=0))
spx.axhline(np.sqrt(ex * tw_rad.betx[0] + ey * tw_rad.betx2[0] + (np.std(p.delta) * tw_rad.dx[0])**2), color='red')
# spx.axhline(np.sqrt(ex_hof * tw.betx[0] + (np.std(p.delta) * tw.dx[0])**2), color='green')

spy = fig. add_subplot(3, 1, 2, sharex=spx)
spy.plot(np.std(mon.y, axis=0))
spy.axhline(np.sqrt(ex * tw_rad.bety1[0] + ey * tw_rad.bety[0] + (np.std(p.delta) * tw_rad.dy[0])**2), color='red')
# spy.axhline(np.sqrt(ey_hof * tw.bety[0] + (np.std(p.delta) * tw.dy[0])**2), color='green')

spz = fig. add_subplot(3, 1, 3, sharex=spx)
spz.plot(np.std(mon.zeta, axis=0))
spz.axhline(np.sqrt(ez * tw_rad.betz0), color='red')

plt.show()