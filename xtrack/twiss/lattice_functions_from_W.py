# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np

from .. import linear_normal_form as lnf


def _get_lattice_functions(Ws, use_full_inverse, s_co):

    # For removal ot thin groups of elements
    i_take = [0]
    for ii in range(1, len(s_co)):
        if s_co[ii] > s_co[ii-1]:
            i_take[-1] = ii-1
            i_take.append(ii)
        else:
            i_take.append(i_take[-1])
    i_take = np.array(i_take)
    _temp_range = np.arange(0, len(s_co), 1, dtype=int)
    mask_replace = _temp_range != i_take
    mask_replace[-1] = False # Force keeping of the last element
    i_replace = _temp_range[mask_replace]
    i_replace_with = i_take[mask_replace]

    # Re normalize eigenvectors (needed when radiation is present)
    nux, nuy, nuzeta = _renormalize_eigenvectors(Ws)

    # Rotate eigenvectors to the Courant-Snyder basis
    phix = np.arctan2(Ws[:, 0, 1], Ws[:, 0, 0])
    phiy = np.arctan2(Ws[:, 2, 3], Ws[:, 2, 2])
    phizeta = np.arctan2(Ws[:, 4, 5], Ws[:, 4, 4])

    v1 = Ws[:, :, 0] + 1j * Ws[:, :, 1]
    v2 = Ws[:, :, 2] + 1j * Ws[:, :, 3]
    v3 = Ws[:, :, 4] + 1j * Ws[:, :, 5]

    for ii in range(6):
        v1[:, ii] *= np.exp(-1j * phix)
        v2[:, ii] *= np.exp(-1j * phiy)
        v3[:, ii] *= np.exp(-1j * phizeta)
    Ws[:, :, 0] = np.real(v1)
    Ws[:, :, 1] = np.imag(v1)
    Ws[:, :, 2] = np.real(v2)
    Ws[:, :, 3] = np.imag(v2)
    Ws[:, :, 4] = np.real(v3)
    Ws[:, :, 5] = np.imag(v3)

    # Computation of twiss parameters
    if use_full_inverse:
        (betx, alfx, gamx, bety, alfy, gamy, bety1, betx2, alfy1, alfx2, gamy1,
        gamx2) = _extract_twiss_parameters_with_inverse(Ws)
    else:
        betx = Ws[:, 0, 0]**2 + Ws[:, 0, 1]**2
        bety = Ws[:, 2, 2]**2 + Ws[:, 2, 3]**2

        gamx = Ws[:, 1, 0]**2 + Ws[:, 1, 1]**2
        gamy = Ws[:, 3, 2]**2 + Ws[:, 3, 3]**2

        alfx = -Ws[:, 0, 0] * Ws[:, 1, 0] - Ws[:, 0, 1] * Ws[:, 1, 1]
        alfy = -Ws[:, 2, 2] * Ws[:, 3, 2] - Ws[:, 2, 3] * Ws[:, 3, 3]

        bety1 = Ws[:, 2, 0]**2 + Ws[:, 2, 1]**2
        betx2 = Ws[:, 0, 2]**2 + Ws[:, 0, 3]**2

        alfx2 = -Ws[:, 0, 2] * Ws[:, 1, 2] - Ws[:, 0, 3] * Ws[:, 1, 3]
        alfy1 = -Ws[:, 2, 0] * Ws[:, 3, 0] - Ws[:, 2, 1] * Ws[:, 3, 1]

        gamx2 = Ws[:, 1, 2]**2 + Ws[:, 1, 3]**2
        gamy1 = Ws[:, 3, 0]**2 + Ws[:, 3, 1]**2

    betx1 = betx
    bety2 = bety

    alfx1 = alfx
    alfy2 = alfy

    gamx1 = gamx
    gamy2 = gamy

    # Edwards-Teng modal Twiss parameters and matrix-equivalent coupling RDTs
    # obtained directly from the phase-gauged normal-form matrix W.
    #
    # The determinant of the principal horizontal Mais-Ripken block is g^4:
    #   beta_x1 * gamma_x1 - alpha_x1**2 = g^4.
    g_squared = np.sqrt(np.maximum(
        betx1 * gamx1 - alfx1**2, 0.0))
    g_edw_teng = np.sqrt(g_squared)

    betx_edw_teng = betx1 / g_squared
    alfx_edw_teng = alfx1 / g_squared
    bety_edw_teng = bety2 / g_squared
    alfy_edw_teng = alfy2 / g_squared

    # Normalize the horizontal components of the complex mode-2 eigenvector.
    # Its phase has already been fixed above by making its y component real
    # and positive.
    sqrt_betx_edw_teng = np.sqrt(betx_edw_teng)
    ux = v2[:, 0] / sqrt_betx_edw_teng
    upx = (
        alfx_edw_teng * v2[:, 0]
        + betx_edw_teng * v2[:, 1]
    ) / sqrt_betx_edw_teng

    f1001 = (upx + 1j * ux) / (4.0 * g_edw_teng)
    f1010 = np.conj(upx - 1j * ux) / (4.0 * g_edw_teng)
    f0110 = np.conj(f1001)
    f0101 = np.conj(f1010)

    temp_phix = phix.copy()
    temp_phiy = phiy.copy()
    temp_phix[i_replace] = temp_phix[i_replace_with]
    temp_phiy[i_replace] = temp_phiy[i_replace_with]

    mux = np.unwrap(temp_phix) / 2 / np.pi
    muy = np.unwrap(temp_phiy) / 2  /np.pi
    muzeta = np.unwrap(phizeta) / 2 / np.pi

    # Crab dispersion
    dx_zeta = (Ws[:, 0, 4] - Ws[:, 0, 5] * Ws[:, 5, 4] / Ws[:, 5, 5]) / (
               Ws[:, 4, 4] - Ws[:, 4, 5] * Ws[:, 5, 4] / Ws[:, 5, 5])
    dpx_zeta = (Ws[:, 1, 4] - Ws[:, 1, 5] * Ws[:, 5, 4] / Ws[:, 5, 5]) / (
                Ws[:, 4, 4] - Ws[:, 4, 5] * Ws[:, 5, 4] / Ws[:, 5, 5])
    dy_zeta = (Ws[:, 2, 4] - Ws[:, 2, 5] * Ws[:, 5, 4] / Ws[:, 5, 5]) / (
                Ws[:, 4, 4] - Ws[:, 4, 5] * Ws[:, 5, 4] / Ws[:, 5, 5])
    dpy_zeta = (Ws[:, 3, 4] - Ws[:, 3, 5] * Ws[:, 5, 4] / Ws[:, 5, 5]) / (
                Ws[:, 4, 4] - Ws[:, 4, 5] * Ws[:, 5, 4] / Ws[:, 5, 5])

    # Dispersion
    dx_pzeta = (Ws[:, 0, 5] - Ws[:, 0, 4] * Ws[:, 4, 5] / Ws[:, 4, 4]) / (
                Ws[:, 5, 5] - Ws[:, 5, 4] * Ws[:, 4, 5] / Ws[:, 4, 4])
    dpx_pzeta = (Ws[:, 1, 5] - Ws[:, 1, 4] * Ws[:, 4, 5] / Ws[:, 4, 4]) / (
                Ws[:, 5, 5] - Ws[:, 5, 4] * Ws[:, 4, 5] / Ws[:, 4, 4])
    dy_pzeta = (Ws[:, 2, 5] - Ws[:, 2, 4] * Ws[:, 4, 5] / Ws[:, 4, 4]) / (
                Ws[:, 5, 5] - Ws[:, 5, 4] * Ws[:, 4, 5] / Ws[:, 4, 4])
    dpy_pzeta = (Ws[:, 3, 5] - Ws[:, 3, 4] * Ws[:, 4, 5] / Ws[:, 4, 4]) / (
                Ws[:, 5, 5] - Ws[:, 5, 4] * Ws[:, 4, 5] / Ws[:, 4, 4])

    mux = mux - mux[0]
    muy = muy - muy[0]
    muzeta = muzeta - muzeta[0]

    res = {
        'betx': betx,
        'bety': bety,
        'alfx': alfx,
        'alfy': alfy,
        'gamx': gamx,
        'gamy': gamy,
        'dx': dx_pzeta,
        'dpx': dpx_pzeta,
        'dy': dy_pzeta,
        'dpy': dpy_pzeta,
        'dx_zeta': dx_zeta,
        'dpx_zeta': dpx_zeta,
        'dy_zeta': dy_zeta,
        'dpy_zeta': dpy_zeta,
        'betx1': betx1,
        'bety1': bety1,
        'betx2': betx2,
        'bety2': bety2,
        'alfx1': alfx1,
        'alfy1': alfy1,
        'alfx2': alfx2,
        'alfy2': alfy2,
        'gamx1': gamx1,
        'gamy1': gamy1,
        'gamx2': gamx2,
        'gamy2': gamy2,
        'betx_edw_teng': betx_edw_teng,
        'alfx_edw_teng': alfx_edw_teng,
        'bety_edw_teng': bety_edw_teng,
        'alfy_edw_teng': alfy_edw_teng,
        'g_edw_teng': g_edw_teng,
        'f1001': f1001,
        'f1010': f1010,
        'f0110': f0110,
        'f0101': f0101,
        'mux': mux,
        'muy': muy,
        'muzeta': muzeta,
        'nux': nux,
        'nuy': nuy,
        'nuzeta': nuzeta,
        'W_matrix': Ws,
        'phix': phix,
        'phiy': phiy,
        'phizeta': phizeta,
    }
    return res, i_replace


def _renormalize_eigenvectors(Ws):
    # Re normalize eigenvectors
    v1 = Ws[:, :, 0] + 1j * Ws[:, :, 1]
    v2 = Ws[:, :, 2] + 1j * Ws[:, :, 3]
    v3 = Ws[:, :, 4] + 1j * Ws[:, :, 5]

    S = lnf.S
    S_v1_imag = v1 * 0.0
    S_v2_imag = v2 * 0.0
    S_v3_imag = v3 * 0.0
    for ii in range(6):
        for jj in range(6):
            if S[ii, jj] !=0:
                S_v1_imag[:, ii] +=  S[ii, jj] * v1.imag[:, jj]
                S_v2_imag[:, ii] +=  S[ii, jj] * v2.imag[:, jj]
                S_v3_imag[:, ii] +=  S[ii, jj] * v3.imag[:, jj]

    nux = np.squeeze(Ws[:, 0, 0]) * (0.0 + 0j)
    nuy = nux * 0.0
    nuzeta = nux * 0.0

    for ii in range(6):
        nux += v1.real[:, ii] * S_v1_imag[:, ii]
        nuy += v2.real[:, ii] * S_v2_imag[:, ii]
        nuzeta += v3.real[:, ii] * S_v3_imag[:, ii]

    nux = np.sqrt(np.abs(nux)) # nux is always positive
    nuy = np.sqrt(np.abs(nuy)) # nuy is always positive
    nuzeta = np.sqrt(np.abs(nuzeta)) # nuzeta is always positive

    for ii in range(6):
        v1[:, ii] /= nux
        v2[:, ii] /= nuy
        v3[:, ii] /= nuzeta

    Ws[:, :, 0] = np.real(v1)
    Ws[:, :, 1] = np.imag(v1)
    Ws[:, :, 2] = np.real(v2)
    Ws[:, :, 3] = np.imag(v2)
    Ws[:, :, 4] = np.real(v3)
    Ws[:, :, 5] = np.imag(v3)

    return nux, nuy, nuzeta


def _extract_twiss_parameters_with_inverse(Ws):

    # From E. Forest, "From tracking code to analysis", Sec 4.1.2 or better
    # https://iopscience.iop.org/article/10.1088/1748-0221/7/07/P07012

    EE = np.zeros(shape=(3, Ws.shape[0], 6, 6), dtype=np.float64)

    for ii in range(3):
        Iii = np.zeros(shape=(6, 6))
        Iii[2*ii, 2*ii] = 1
        Iii[2*ii+1, 2*ii+1] = 1
        Sii = lnf.S @ Iii

        Ws_inv = np.linalg.inv(Ws)

        EE[ii, :, :, :] = - Ws @ Sii @ Ws_inv @ lnf.S

    betx = EE[0, :, 0, 0]
    bety = EE[1, :, 2, 2]
    alfx = -EE[0, :, 0, 1]
    alfy = -EE[1, :, 2, 3]
    gamx = EE[0, :, 1, 1]
    gamy = EE[1, :, 3, 3]

    bety1 = np.abs(EE[0, :, 2, 2])
    betx2 = np.abs(EE[1, :, 0, 0])

    alfy1 = -EE[0, :, 2, 3]
    alfx2 = -EE[1, :, 0, 1]

    gamy1 = EE[0, :, 3, 3]
    gamx2 = EE[1, :, 1, 1]

    sign_x = np.sign(betx)
    sign_y = np.sign(bety)
    betx *= sign_x
    alfx *= sign_x
    gamx *= sign_x
    bety *= sign_y
    alfy *= sign_y
    gamy *= sign_y

    return betx, alfx, gamx, bety, alfy, gamy, bety1, betx2, alfy1, alfx2, gamy1, gamx2
