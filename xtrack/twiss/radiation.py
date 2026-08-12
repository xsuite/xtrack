# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np
from scipy.constants import c as clight
from scipy.constants import epsilon_0
from scipy.constants import e as qe
from scipy.constants import electron_volt
from scipy.constants import hbar

from .. import linear_normal_form as lnf
from ..table import Table
from .beam_covariance import _build_sigma_table
from .trajectory_curvatures import _get_trajectory_curvatures

import xtrack as xt  # To avoid circular imports


def _get_eneloss_and_damping_rates(particle_on_co, R_matrix,
                                       px_co, py_co, ptau_co, W_matrix,
                                       t_rev0, line, radiation_method):
    diff_ptau = np.diff(ptau_co)
    mask_loss = diff_ptau < 0
    eloss_turn = -sum(diff_ptau[mask_loss]) * particle_on_co._xobject.p0c[0]

    # Get eigenvalues
    w0, v0 = np.linalg.eig(R_matrix)

    # Sort eigenvalues
    modes = lnf.sort_modes(v0, w0)
    eigenvals = np.array([w0[ii] for ii in modes])

    # Damping constants and partition numbers
    energy0 = particle_on_co.mass0 * particle_on_co._xobject.gamma0[0]

    damping_constants_turns = -np.log(np.abs(eigenvals))
    damping_constants_s = damping_constants_turns / t_rev0

    # https://cds.cern.ch/record/175614 , Eq. 4.24
    partition_numbers = (
        damping_constants_turns * 2
        / (-np.sum(diff_ptau[mask_loss] / (1 + ptau_co[:-1][mask_loss]))))

    eneloss_damp_res = {
        'eneloss_turn': eloss_turn, # deprecated
        'energy_loss': eloss_turn,
        'damping_constants_turns': damping_constants_turns,
        'damping_constants_s':damping_constants_s,
        'partition_numbers': partition_numbers,
    }

    return eneloss_damp_res


def _extract_sr_distribution_properties(twiss_res):

    radiation_flag = twiss_res['radiation_flag']
    if np.any(
            (radiation_flag == 2)
            | (radiation_flag == 3)):
        raise ValueError('Incompatible radiation flag')

    hx, hy, kappa0_x, kappa0_y = _get_trajectory_curvatures(twiss_res)
    hh = np.sqrt(hx**2 + hy**2)

    ptau_co = twiss_res['ptau']
    dl = twiss_res['length'] * (twiss_res['radiation_flag'] == 1)

    pco = twiss_res['particle_on_co']
    mass0 = pco.mass0
    q0 = pco.q0
    gamma0 = pco._xobject.gamma0[0]
    beta0 = pco._xobject.beta0[0]

    gamma = gamma0 * (1 + beta0 * ptau_co)

    mass0_kg = mass0 / clight**2 * qe
    q_coul = q0 * qe
    B_T = hh * mass0_kg * clight * gamma0 / np.abs(q_coul)
    r0_m = q_coul**2/(4*np.pi*epsilon_0*mass0_kg*clight**2)
    E_crit_J = 3 * np.abs(q_coul) * hbar * gamma**2 * B_T / (2 * mass0_kg)
    n_dot = 60 / 72 * np.sqrt(3) * r0_m * clight * np.abs(q_coul) * B_T / hbar
    E_sq_ave_J = 11 / 27 * E_crit_J**2
    E_ave_J = 8 * np.sqrt(3) / 45 * E_crit_J
    E0_J = mass0_kg * clight**2 * gamma0

    n_dot_delta_kick_sq_ave = n_dot * E_sq_ave_J / E0_J**2

    res = {
        'B_T': B_T,
        'hx': hx, 'hy': hy,
        'h0x': kappa0_x, 'h0y': kappa0_y,
        'E_crit_J': E_crit_J, 'n_dot': n_dot,
        'E_sq_ave_J': E_sq_ave_J, 'E_ave_J': E_ave_J,
        'n_dot_delta_kick_sq_ave': n_dot_delta_kick_sq_ave,
        'dl_radiation': dl,
    }

    return res


def _get_equilibrium_emittance_kick_as_co(twiss_res,
                                  damping_constants_turns,
                                  radiation_method):

    assert radiation_method == 'kick_as_co'

    sr_distrib_properties = _extract_sr_distribution_properties(twiss_res)

    pco = twiss_res['particle_on_co']
    beta0 = pco._xobject.beta0[0]
    gamma0 = pco._xobject.gamma0[0]

    kin_px_co = twiss_res['kin_px']
    kin_py_co = twiss_res['kin_py']
    ptau_co = twiss_res['ptau']
    W_matrix = twiss_res['W_matrix']

    n_dot_delta_kick_sq_ave = sr_distrib_properties['n_dot_delta_kick_sq_ave'][:-1]
    dl = sr_distrib_properties['dl_radiation'][:-1]

    px_left = kin_px_co[:-1]
    px_right = kin_px_co[1:]
    py_left = kin_py_co[:-1]
    py_right = kin_py_co[1:]
    one_pl_del_left = (1 + ptau_co[:-1]) # Assuming ultrarelativistic
    one_pl_del_right = (1 + ptau_co[1:]) # Assuming ultrarelativistic
    W_left = W_matrix[:-1, :, :]
    W_right = W_matrix[1:, :, :]

    a11_left = np.squeeze(W_left[:, 0, 0])
    a13_left = np.squeeze(W_left[:, 2, 0])
    a15_left = np.squeeze(W_left[:, 4, 0])
    b11_left = np.squeeze(W_left[:, 0, 1])
    b13_left = np.squeeze(W_left[:, 2, 1])
    b15_left = np.squeeze(W_left[:, 4, 1])

    a11_right = np.squeeze(W_right[:, 0, 0])
    a13_right = np.squeeze(W_right[:, 2, 0])
    a15_right = np.squeeze(W_right[:, 4, 0])
    b11_right = np.squeeze(W_right[:, 0, 1])
    b13_right = np.squeeze(W_right[:, 2, 1])
    b15_right = np.squeeze(W_right[:, 4, 1])

    a21_left = np.squeeze(W_left[:, 0, 2])
    a23_left = np.squeeze(W_left[:, 2, 2])
    a25_left = np.squeeze(W_left[:, 4, 2])
    b21_left = np.squeeze(W_left[:, 0, 3])
    b23_left = np.squeeze(W_left[:, 2, 3])
    b25_left = np.squeeze(W_left[:, 4, 3])

    a21_right = np.squeeze(W_right[:, 0, 2])
    a23_right = np.squeeze(W_right[:, 2, 2])
    a25_right = np.squeeze(W_right[:, 4, 2])
    b21_right = np.squeeze(W_right[:, 0, 3])
    b23_right = np.squeeze(W_right[:, 2, 3])
    b25_right = np.squeeze(W_right[:, 4, 3])

    a31_left = np.squeeze(W_left[:, 0, 4])
    a33_left = np.squeeze(W_left[:, 2, 4])
    a35_left = np.squeeze(W_left[:, 4, 4])
    b31_left = np.squeeze(W_left[:, 0, 5])
    b33_left = np.squeeze(W_left[:, 2, 5])
    b35_left = np.squeeze(W_left[:, 4, 5])

    a31_right = np.squeeze(W_right[:, 0, 4])
    a33_right = np.squeeze(W_right[:, 2, 4])
    a35_right = np.squeeze(W_right[:, 4, 4])
    b31_right = np.squeeze(W_right[:, 0, 5])
    b33_right = np.squeeze(W_right[:, 2, 5])
    b35_right = np.squeeze(W_right[:, 4, 5])

    Kx_left = (a11_left * px_left + a13_left * py_left) / one_pl_del_left + a15_left
    Kpx_left = (b11_left * px_left + b13_left * py_left) / one_pl_del_left + b15_left
    Ky_left = (a21_left * px_left + a23_left * py_left) / one_pl_del_left + a25_left
    Kpy_left = (b21_left * px_left + b23_left * py_left) / one_pl_del_left + b25_left
    Kz_left = (a31_left * px_left + a33_left * py_left) / one_pl_del_left + a35_left
    Kpz_left = (b31_left * px_left + b33_left * py_left) / one_pl_del_left + b35_left

    Kx_right = (a11_right * px_right + a13_right * py_right) / one_pl_del_right + a15_right
    Kpx_right = (b11_right * px_right + b13_right * py_right) / one_pl_del_right + b15_right
    Ky_right = (a21_right * px_right + a23_right * py_right) / one_pl_del_right + a25_right
    Kpy_right = (b21_right * px_right + b23_right * py_right) / one_pl_del_right + b25_right
    Kz_right = (a31_right * px_right + a33_right * py_right) / one_pl_del_right + a35_right
    Kpz_right = (b31_right * px_right + b33_right * py_right) / one_pl_del_right + b35_right

    Kx_sq = 0.5 * (Kx_left**2 + Kx_right**2)
    Kpx_sq = 0.5 * (Kpx_left**2 + Kpx_right**2)
    Ky_sq = 0.5 * (Ky_left**2 + Ky_right**2)
    Kpy_sq = 0.5 * (Kpy_left**2 + Kpy_right**2)
    Kz_sq = 0.5 * (Kz_left**2 + Kz_right**2)
    Kpz_sq = 0.5 * (Kpz_left**2 + Kpz_right**2)

    eq_gemitt_x = 1 / (4 * clight * damping_constants_turns[0]) * np.sum(
                        (Kx_sq + Kpx_sq) * n_dot_delta_kick_sq_ave * dl)
    eq_gemitt_y = 1 / (4 * clight * damping_constants_turns[1]) * np.sum(
                        (Ky_sq + Kpy_sq) * n_dot_delta_kick_sq_ave * dl)
    eq_gemitt_zeta = 1 / (4 * clight * damping_constants_turns[2]) * np.sum(
                        (Kz_sq + Kpz_sq) * n_dot_delta_kick_sq_ave * dl)

    eq_nemitt_x = float(eq_gemitt_x * (beta0 * gamma0))
    eq_nemitt_y = float(eq_gemitt_y * (beta0 * gamma0))
    eq_nemitt_zeta = float(eq_gemitt_zeta * (beta0 * gamma0))

    res = {
        'eq_gemitt_x': eq_gemitt_x,
        'eq_gemitt_y': eq_gemitt_y,
        'eq_gemitt_zeta': eq_gemitt_zeta,
        'eq_nemitt_x': eq_nemitt_x,
        'eq_nemitt_y': eq_nemitt_y,
        'eq_nemitt_zeta': eq_nemitt_zeta,
        'dl_radiation': dl,
        'n_dot_delta_kick_sq_ave': n_dot_delta_kick_sq_ave,
    }

    return res


def _get_equilibrium_emittance_full(twiss_res, R_matrix_ebe,
                                        radiation_method):

    kin_px_co = twiss_res['kin_px']
    kin_py_co = twiss_res['kin_py']
    ptau_co = twiss_res['ptau']

    sr_distrib_properties = _extract_sr_distribution_properties(twiss_res)

    n_dot_delta_kick_sq_ave = sr_distrib_properties['n_dot_delta_kick_sq_ave'][:-1]
    dl = sr_distrib_properties['dl_radiation'][:-1]

    assert radiation_method == 'full'

    d_delta_sq_ave = n_dot_delta_kick_sq_ave * dl / clight

    # Going to x', y'
    RR_ebe = R_matrix_ebe
    delta = ptau_co # ultrarelativistic approximation

    TT = RR_ebe * 0.
    TT[:, 0, 0] = 1
    TT[:, 1, 1] = (1 - delta)
    TT[:, 1, 5] = -kin_px_co
    TT[:, 2, 2] = 1
    TT[:, 3, 3] = (1 - delta)
    TT[:, 3, 5] = -kin_py_co
    TT[:, 4, 4] = 1
    TT[:, 5, 5] = 1

    TTinv = np.linalg.inv(TT)
    TTinv0 = TTinv.copy()
    for ii in range(6):
        for jj in range(6):
            TTinv0[:, ii, jj] = TTinv[0, ii, jj]

    RR_ebe_hat = TT @ RR_ebe @ TTinv0
    RR = RR_ebe_hat[-1, :, :]

    lnf = xt.linear_normal_form
    WW, _, Rot, lam_eig = lnf.get_linear_normal_form(RR)
    DSigma = np.zeros_like(RR_ebe_hat)

    # The following is needed if RR is in px, py instead of x', y'
    # DSigma[:-1, 1, 1] = (d_delta_sq_ave * 0.5 * (px_co[:-1]**2 + px_co[1:]**2)
    #                                             / (ptau_co[:-1] + 1)**2)
    # DSigma[:-1, 3, 3] = (d_delta_sq_ave * 0.5 * (py_co[:-1]**2 + py_co[1:]**2)
    #                                             / (ptau_co[:-1] + 1)**2)

    # DSigma[:-1, 1, 5] = (d_delta_sq_ave * 0.5 * (px_co[:-1] + px_co[1:])
    #                                             / (ptau_co[:-1] + 1))
    # DSigma[:-1, 5, 1] = (d_delta_sq_ave * 0.5 * (px_co[:-1] + px_co[1:])
    #                                             / (ptau_co[:-1] + 1))

    # DSigma[:-1, 3, 5] = (d_delta_sq_ave * 0.5 * (py_co[:-1] + py_co[1:])
    #                                              / (ptau_co[:-1] + 1))
    # DSigma[:-1, 5, 3] = (d_delta_sq_ave * 0.5 * (py_co[:-1] + py_co[1:])
    #                                              / (ptau_co[:-1] + 1))

    DSigma[:-1, 5, 5] = d_delta_sq_ave

    RR_ebe_hat_inv = np.linalg.inv(RR_ebe_hat)

    DSigma0 = np.zeros((6, 6))

    n_calc = d_delta_sq_ave.shape[0]
    for ii in range(n_calc):
        if d_delta_sq_ave[ii] > 0:
            DSigma0 += RR_ebe_hat_inv[ii, :, :] @ DSigma[ii, :, :] @ RR_ebe_hat_inv[ii, :, :].T

    CC_split, _, RRR, reig = lnf.get_linear_normal_form(Rot)
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

    eq_gemitt_x = EE_norm[0, 1]/(1 - np.abs(lam_eig[0])**2)
    eq_gemitt_y = EE_norm[2, 3]/(1 - np.abs(lam_eig[1])**2)
    eq_gemitt_zeta = EE_norm[4, 5]/(1 - np.abs(lam_eig[2])**2)

    beta0 = twiss_res.particle_on_co._xobject.beta0[0]
    gamma0 = twiss_res.particle_on_co._xobject.gamma0[0]

    eq_nemitt_x = float(eq_gemitt_x * (beta0 * gamma0))
    eq_nemitt_y = float(eq_gemitt_y * (beta0 * gamma0))
    eq_nemitt_zeta = float(eq_gemitt_zeta * (beta0 * gamma0))

    Sigma_norm = np.zeros_like(EE_norm, dtype=complex)
    for ii in range(6):
        for jj in range(6):
            Sigma_norm[ii, jj] = EE_norm[ii, jj]/(1 - lam_eig_full[ii, ii]*lam_eig_full[jj, jj])

    Sigma_at_start = (BB @ Sigma_norm @ BB.T).real

    Sigma = RR_ebe @ Sigma_at_start @ np.transpose(RR_ebe, axes=(0,2,1))

    eq_sigma_tab = _build_sigma_table(Sigma=Sigma, s=None, name=twiss_res['name'],)

    res = {
        'eq_gemitt_x': eq_gemitt_x,
        'eq_gemitt_y': eq_gemitt_y,
        'eq_gemitt_zeta': eq_gemitt_zeta,
        'eq_nemitt_x': eq_nemitt_x,
        'eq_nemitt_y': eq_nemitt_y,
        'eq_nemitt_zeta': eq_nemitt_zeta,
        'eq_beam_covariance_matrix': eq_sigma_tab,
        'dl_radiation': dl,
        'n_dot_delta_kick_sq_ave': n_dot_delta_kick_sq_ave,
        'hx_rad': sr_distrib_properties['hx'],
        'hy_rad': sr_distrib_properties['hy'],
        'h0x_rad': sr_distrib_properties['h0x'],
        'h0y_rad': sr_distrib_properties['h0y'],
    }

    return res


def _compute_radiation_integrals(twiss_res):

    kin_px = twiss_res['kin_px']
    kin_py = twiss_res['kin_py']
    delta = twiss_res['delta']
    length = twiss_res['length']

    betx = twiss_res['betx']             # Twiss beta function x
    alfx = twiss_res['alfx']             # Twiss alpha x
    gamx = twiss_res['gamx']             # Twiss gamma x
    bety = twiss_res['bety']             # Twiss beta function y
    alfy = twiss_res['alfy']             # Twiss alpha y
    gamy = twiss_res['gamy']             # Twiss gamma y
    dx = twiss_res['dx']                 # Dispersion x
    dy = twiss_res['dy']                 # Dispersion y
    dpx = twiss_res['dpx']               # Dispersion px
    dpy = twiss_res['dpy']               # Dispersion py

    mass0 = twiss_res.particle_on_co.mass0
    r0 = twiss_res.particle_on_co.get_classical_particle_radius0()
    gamma0 = twiss_res.particle_on_co.gamma0[0]

    dxprime = dpx * (1 - delta) - kin_px
    dyprime = dpy * (1 - delta) - kin_py

    kappa_x, kappa_y, kappa0_x, kappa0_y = _get_trajectory_curvatures(twiss_res)
    kappa = np.sqrt(kappa_x**2 + kappa_y**2)
    kappa0 = np.sqrt(kappa0_x**2 + kappa0_y**2)

    # quadrupole gradient
    mask = length != 0
    k1 = 0 * length
    k1[mask] = twiss_res.k1l[mask] / length[mask]

    # Curly H
    Hx_rad = gamx * dx**2 + 2*alfx * dx * dxprime + betx * dxprime**2
    Hy_rad = gamy * dy**2 + 2*alfy * dy * dyprime + bety * dyprime**2

    # Integrands
    i1x_integrand = kappa0_x * dx
    i1y_integrand = kappa0_y * dy

    i2_integrand = kappa * kappa

    i3_integrand = np.abs(kappa * kappa * kappa)

    i4x_integrand = dx * (kappa0_x * kappa**2 + 2 * k1 * kappa_x)
    i4y_integrand = dy * (kappa0_y * kappa**2 - 2 * k1 * kappa_y)
    i4_integrand = i4x_integrand + i4y_integrand


    i5x_integrand = np.abs(kappa * kappa * kappa) * Hx_rad
    i5y_integrand = np.abs(kappa * kappa * kappa) * Hy_rad

    # Integrate
    i1x = np.sum(i1x_integrand * length)
    i1y = np.sum(i1y_integrand * length)
    i2 = np.sum(i2_integrand * length)
    i3 = np.sum(i3_integrand * length)
    i4 = np.sum(i4_integrand * length)
    i4x = np.sum(i4x_integrand * length)
    i4y = np.sum(i4y_integrand * length)
    i5x = np.sum(i5x_integrand * length)
    i5y = np.sum(i5y_integrand * length)

    # Emittances
    eq_gemitt_x = (55/(32 * 3**(1/2)) * hbar / electron_volt * clight
                / mass0 * gamma0**2 * i5x / (i2 - i4x))
    eq_gemitt_y = (55/(32 * 3**(1/2)) * hbar / electron_volt * clight
                / mass0 * gamma0**2 * i5y / (i2 - i4y))
    energy0 = twiss_res.particle_on_co.energy0[0]
    energy_loss = 2 / 3 * r0 * energy0 * gamma0**3 * i2
    sigma_delta = np.sqrt(55 * np.sqrt(3) / 96
                          * hbar / electron_volt * clight
                          / mass0 * gamma0**2 * i3 / (2 * i2 + i4))

    # Damping constants
    damping_constant_x_s = r0/3 * gamma0**3 * clight/twiss_res.line_length * (i2 - i4x)
    damping_constant_y_s = r0/3 * gamma0**3 * clight/twiss_res.line_length * (i2 - i4y)
    damping_constant_zeta_s = r0/3 * gamma0**3 * clight/twiss_res.line_length * (2*i2 + i4)

    # Damping partition numbers:
    J_x = 1 - i4x / i2
    J_y = 1 - i4y / i2
    J_zeta = 2 + i4 / i2

    # Velocity direction (for spin)
    ps = np.sqrt((1 + delta)**2 - kin_px**2 - kin_py**2)
    xp = kin_px / ps
    yp = kin_py / ps
    tempv = np.sqrt(xp**2 + yp**2 + 1)
    iv_x = xp / tempv
    iv_y = yp / tempv
    iv_z = 1 / tempv

    cols = {
        'rad_int_curly_hx': Hx_rad,
        'rad_int_curly_hy': Hy_rad,
        'rad_int_i1x_integrand': i1x_integrand,
        'rad_int_i1y_integrand': i1y_integrand,
        'rad_int_l2_integrand': i2_integrand,
        'rad_int_i3_integrand': i3_integrand,
        'rad_int_i4_integrand': i4_integrand,
        'rad_int_i4x_integrand': i4x_integrand,
        'rad_int_i4y_integrand': i4y_integrand,
        'rad_int_i5x_integrand': i5x_integrand,
        'rad_int_i5y_integrand': i5y_integrand,
        'rad_int_kappa0_x': kappa0_x,
        'rad_int_kappa0_y': kappa0_y,
        'rad_int_kappa0': kappa0,
        'rad_int_kappa_x': kappa_x,
        'rad_int_kappa_y': kappa_y,
        'rad_int_kappa': kappa,
        'rad_int_iv_x': iv_x,
        'rad_int_iv_y': iv_y,
        'rad_int_iv_z': iv_z,
    }

    scalars = {
        'rad_int_i1x': i1x,
        'rad_int_i1y': i1y,
        'rad_int_i2': i2,
        'rad_int_i3': i3,
        'rad_int_i4': i4,
        'rad_int_i4x': i4x,
        'rad_int_i4y': i4y,
        'rad_int_i5x': i5x,
        'rad_int_i5y': i5y,
        'rad_int_eq_gemitt_x': eq_gemitt_x,
        'rad_int_eq_gemitt_y': eq_gemitt_y,
        'rad_int_energy_loss': energy_loss,
        'rad_int_sigma_delta': sigma_delta,
        'rad_int_damping_constant_x_s': damping_constant_x_s,
        'rad_int_damping_constant_y_s': damping_constant_y_s,
        'rad_int_damping_constant_zeta_s': damping_constant_zeta_s,
        'rad_int_partition_number_x': J_x,
        'rad_int_partition_number_y': J_y,
        'rad_int_partition_number_zeta': J_zeta,
    }

    out = Table({'name': twiss_res.name, 's': twiss_res.s, 'length': length} | cols)
    out._data.update(scalars)

    return out
