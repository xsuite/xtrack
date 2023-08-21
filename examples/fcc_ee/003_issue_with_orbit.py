import numpy as np
import xtrack as xt

line = xt.Line.from_json('fccee_p_ring_thin_wig.json')
line.build_tracker()
varvals = line._xdeps_vref._owner.copy()

# Add longer wigglers
section_list = [
    ('qrfr1.1_entry', 'qrfl1.1_exit'),
    # ('qrfr1.2_entry', 'qrfl1.2_exit'), # taken by the RF
    # ('qrfr1.3_entry', 'qrfl1.3_exit'),
    # ('qrfr1.4_entry', 'qrfl1.4_exit'),
]


n_slices = 10
line.vars['k0l_long_wig'] = 0
line.vars['wig_drift_fraction'] = 0.7
for iss, section in enumerate(section_list):

    tt = line.get_table()
    tt_insertion = tt.rows[section[0]: section[1]]
    mask_drift_q = tt_insertion.mask['drift_q.*']
    tt_wig_drifts = tt_insertion.rows[~mask_drift_q].rows['drift_.*']

    for jj, nn in enumerate(tt_wig_drifts.name):
        if jj > 1: break
        s_start_wig = tt['s', nn]
        s_end_wig = tt['s', nn] + line[nn].length

        tag_wig = f'w{iss}_{jj}'

        n_kicks = 4
        s_wig_kicks = np.linspace(s_start_wig, s_end_wig, n_kicks)
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
                line.element_refs[nn_plus].length = half_period * (1 - line.vars['wig_drift_fraction']) / n_slices
                line.element_refs[nn_minus].length = half_period * (1 - line.vars['wig_drift_fraction']) / n_slices
                if ii == 0:
                    line.element_refs[nn_plus].ksl[0] = line.vars[kk_wig] / n_slices * 0.5
                if ii == len(s_wig_plus) - 1:
                    line.element_refs[nn_minus].ksl[0] = -line.vars[kk_wig] / n_slices * 0.5

line.to_json('fccee_p_ring_thin_wig_long.json')

line = line.from_json('fccee_p_ring_thin_wig_long.json')
line.build_tracker()

line.vars['k0l_long_wig'] = 8e-4

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
plt.show()