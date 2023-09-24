import numpy as np
import xtrack as xt

num_particles_test = 300

# fname = 'fccee_z'; gemitt_y_target = 1.4e-12; n_turns_track_test = 3000
# fname = 'fccee_w'; gemitt_y_target = 2.2e-12; n_turns_track_test = 1000
fname = 'fccee_h'; gemitt_y_target = 1.4e-12; n_turns_track_test = 400
fname = 'fccee_t'; gemitt_y_target = 2e-12; n_turns_track_test = 400


line = xt.Line.from_json(fname + '_thin.json')

line.cycle('qrdr2.3_entry', inplace=True)

# Add monitor in a dispersion-free place out of crab waist
monitor = xt.ParticlesMonitor(num_particles=num_particles_test,
                              start_at_turn=0, stop_at_turn=n_turns_track_test)
line.insert_element(element=monitor, name='monitor', index='qrdr2.3_entry')

line.build_tracker()

tw_no_rad = line.twiss(method='4d')

line.configure_radiation(model='mean')
line.compensate_radiation_energy_loss()

tw_rad_wig_off = line.twiss(eneloss_and_damping=True)

line.vars['on_wiggler_v'] = 0.1
line.compensate_radiation_energy_loss()
opt = line.match(
    solve=False,
    eneloss_and_damping=True,
    compensate_radiation_energy_loss=True,
    targets=[
        xt.Target(eq_gemitt_y=gemitt_y_target, tol=1e-15, optimize_log=True)],
    vary=xt.Vary('on_wiggler_v', step=0.01, limits=(0.1, 2))
)

opt.solve()

tw_rad = line.twiss(eneloss_and_damping=True)

ex = tw_rad.eq_gemitt_x
ey = tw_rad.eq_gemitt_y
ez = tw_rad.eq_gemitt_zeta

line.configure_radiation(model='quantum')
p = line.build_particles(num_particles=num_particles_test)
line.track(p, num_turns=n_turns_track_test, turn_by_turn_monitor=True, time=True)
mon_at_start = line.record_last_track
print(f'Tracking time: {line.time_last_track}')

from scipy.constants import c as clight

line.configure_radiation(model='mean')
tw_rad2 = line.twiss(eneloss_and_damping=True, method='6d',
                     radiation_method='full',
                     compute_R_element_by_element=True)

d_delta_sq_ave = tw_rad.n_dot_delta_kick_sq_ave * tw_rad.dl_radiation /clight
RR_ebe = tw_rad2.R_matrix_ebe.copy()

# # Going to x', y'
# for jj in range(6):
#     RR_ebe[:, 1, jj] /= (1 + tw_rad2.delta)
#     RR_ebe[:, 3, jj] /= (1 + tw_rad2.delta)

# for ii in range(6):
#     RR_ebe[:, ii, 1] *= (1 + tw_rad2.delta)
#     RR_ebe[:, ii, 3] *= (1 + tw_rad2.delta)

# RR_ebe[:, 1, 5] += tw_rad2.px
# RR_ebe[:, 3, 5] += tw_rad2.py

lnf = xt.linear_normal_form
RR = RR_ebe[-1, :, :]
WW, _, Rot, lam_eig = lnf.compute_linear_normal_form(RR)
DSigma = np.zeros_like(RR_ebe)

DSigma[:-1, 1, 1] = (d_delta_sq_ave * 0.5 * (tw_rad2.px[:-1]**2 + tw_rad2.px[1:]**2)
                                            / (tw_rad2.delta[:-1] + 1)**2)
DSigma[:-1, 3, 3] = (d_delta_sq_ave * 0.5 * (tw_rad2.py[:-1]**2 + tw_rad2.py[1:]**2)
                                            / (tw_rad2.delta[:-1] + 1)**2)
DSigma[:-1, 3, 5] = (d_delta_sq_ave * 0.5 * (tw_rad2.py[:-1] + tw_rad2.py[1:])
                                             / (tw_rad2.delta[:-1] + 1))
DSigma[:-1, 5, 3] = (d_delta_sq_ave * 0.5 * (tw_rad2.py[:-1] + tw_rad2.py[1:])
                                             / (tw_rad2.delta[:-1] + 1))
DSigma[:-1, 5, 5] = d_delta_sq_ave

RR_ebe_inv = np.linalg.inv(RR_ebe)

DSigma0 = np.zeros((6, 6))

n_calc = d_delta_sq_ave.shape[0]
for ii in range(n_calc):
    print(f'{ii}/{n_calc}    ', end='\r', flush=True)
    if d_delta_sq_ave[ii] > 0:
        DSigma0 += RR_ebe_inv[ii, :, :] @ DSigma[ii, :, :] @ RR_ebe_inv[ii, :, :].T


CC_split, _, RRR, reig = lnf.compute_linear_normal_form(Rot)
reig_full = np.zeros_like(Rot, dtype=complex)
reig_full[0, 0] = reig[0]
reig_full[1, 1] = reig[0].conjugate()
reig_full[2, 2] = reig[1]
reig_full[3, 3] = reig[1].conjugate()
reig_full[4, 4] = reig[2]
reig_full[5, 5] = reig[2].conjugate()

lam_eig_full = np.zeros_like(reig_full, dtype=complex)
lam_eig_full[0] = lam_eig[0]
lam_eig_full[1] = lam_eig[0].conjugate()
lam_eig_full[2] = lam_eig[1]
lam_eig_full[3] = lam_eig[1].conjugate()
lam_eig_full[4] = lam_eig[2]
lam_eig_full[5] = lam_eig[2].conjugate()

CC = np.zeros_like(CC_split, dtype=complex)
CC[:, 0] = 0.5*np.sqrt(2)*(CC_split[:, 0] + 1j*CC_split[:, 1])
CC[:, 1] = 0.5*np.sqrt(2)*(CC_split[:, 0] - 1j*CC_split[:, 1])
CC[:, 2] = 0.5*np.sqrt(2)*(CC_split[:, 2] + 1j*CC_split[:, 3])
CC[:, 3] = 0.5*np.sqrt(2)*(CC_split[:, 2] - 1j*CC_split[:, 3])
CC[:, 4] = 0.5*np.sqrt(2)*(CC_split[:, 4] + 1j*CC_split[:, 5])
CC[:, 5] = 0.5*np.sqrt(2)*(CC_split[:, 4] - 1j*CC_split[:, 5])

BB = WW @ CC

BB_inv = np.linalg.inv(BB)

EE_norm = (BB_inv @ DSigma0 @ BB_inv.T).real

ex_forest = EE_norm[0, 1]/(1 - np.abs(lam_eig[0])**2)
ey_forest = EE_norm[2, 3]/(1 - np.abs(lam_eig[1])**2)
ez_forest = EE_norm[4, 5]/(1 - np.abs(lam_eig[2])**2)

Sigma_norm = np.zeros_like(EE_norm, dtype=complex)
for ii in range(6):
    for jj in range(6):
        Sigma_norm[ii, jj] = EE_norm[ii, jj]/(1 - lam_eig_full[ii, ii]*lam_eig_full[jj, jj])

Sigma = (BB @ Sigma_norm @ BB.T).real

import matplotlib.pyplot as plt
plt.close('all')
for ii, (mon, element_mon, label) in enumerate(
                            [(mon_at_start, 0, 'inside crab waist'),
                            #  (monitor, 'monitor', 'outside crab waist')
                             ]):

    betx = tw_rad['betx', element_mon]
    bety = tw_rad['bety', element_mon]
    betx2 = tw_rad['betx2', element_mon]
    bety1 = tw_rad['bety1', element_mon]
    dx = tw_rad['dx', element_mon]
    dy = tw_rad['dy', element_mon]

    sigma_tab = tw_rad.get_beam_covariance(gemitt_x=ex_forest,
                                          gemitt_y=ey_forest,
                                          gemitt_zeta=ez_forest)

    fig = plt.figure(ii + 1, figsize=(6.4, 4.8*1.3))
    spx = fig. add_subplot(3, 1, 1)
    spx.plot(np.std(mon.x, axis=0), label='track')
    spx.axhline(
        # sigma_tab['sigma_x', element_mon],
        np.sqrt(Sigma[0, 0]),
        color='red', label='twiss')
    spx.legend(loc='lower right')
    spx.set_ylabel(r'$\sigma_{x}$ [m]')
    spx.set_ylim(bottom=0)

    spy = fig. add_subplot(3, 1, 2, sharex=spx)
    spy.plot(np.std(mon.y, axis=0), label='track')
    spy.axhline(
        # sigma_tab['sigma_y', element_mon],
        np.sqrt(Sigma[2, 2]),
        color='red', label='twiss')
    spy.set_ylabel(r'$\sigma_{y}$ [m]')
    spy.set_ylim(bottom=0)

    spz = fig. add_subplot(3, 1, 3, sharex=spx)
    spz.plot(np.std(mon.zeta, axis=0))
    spz.axhline(
        # sigma_tab['sigma_zeta', element_mon],
        np.sqrt(Sigma[4, 4]),
        color='red')
    spz.set_ylabel(r'$\sigma_{z}$ [m]')
    spz.set_ylim(bottom=0)

    plt.suptitle(f'{fname} - ' r'$\varepsilon_y$ = ' f'{ey*1e12:.6f} pm - {label}')

plt.show()