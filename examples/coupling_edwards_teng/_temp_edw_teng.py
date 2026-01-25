import numpy as np
import xtrack as xt



def _compute_edwards_teng_initial(RR):

    AA = RR[:2, :2]
    BB = RR[:2, 2:4]
    CC = RR[2:4, :2]
    DD = RR[2:4, 2:4]

    if np.linalg.norm(BB) < 1e-10 and np.linalg.norm(CC) < 1e-10:
        RR_ET0 = np.zeros((2, 2))
    else:
        tr = np.linalg.trace
        b_pl_c = CC + _conj_mat(BB)
        det_bc = np.linalg.det(b_pl_c)
        tr_a_m_tr_d = tr(AA) - tr(DD)
        coeff = - (0.5 * tr_a_m_tr_d
            + np.sign(det_bc) * np.sqrt(det_bc + 0.25 * tr_a_m_tr_d**2))
        RR_ET0 = 1/coeff * b_pl_c

    EE = AA - BB@RR_ET0
    FF = DD + RR_ET0@BB

    quarter = 0.25
    two = 2.0

    sinmu2 = -EE[0,1]*EE[1,0] - quarter*(EE[0,0] - EE[1,1])**2
    sinmux = np.sign(EE[0,1]) * np.sqrt(abs(sinmu2))
    betx0 = EE[0,1] / sinmux
    alfx0 = (EE[0,0] - EE[1,1]) / (two * sinmux)

    sinmu2 = -FF[0,1]*FF[1,0] - quarter*(FF[0,0] - FF[1,1])**2
    sinmuy = np.sign(FF[0,1]) * np.sqrt(abs(sinmu2))
    bety0 = FF[0,1] / sinmuy
    alfy0 = (FF[0,0] - FF[1,1]) / (two * sinmuy)

    edw_teng_init = {
        'RR_ET0': RR_ET0,
        'betx0': betx0,
        'alfx0': alfx0,
        'bety0': bety0,
        'alfy0': alfy0
    }

    return edw_teng_init

def _conj_mat(mm):
    a = mm[0,0]
    b = mm[0,1]
    c = mm[1,0]
    d = mm[1,1]
    return np.array([[d, -b], [-c, a]])

def _propagate_edwards_teng(WW, mux, muy, RR_ET0, betx0, alfx0, bety0, alfy0):

    lnf = xt.linear_normal_form
    SS2D = lnf.S[:2, :2]

    RR_ET = RR_ET0.copy()

    n_elem = len(mux)
    betx = np.zeros(n_elem)
    alfx = np.zeros(n_elem)
    bety = np.zeros(n_elem)
    alfy = np.zeros(n_elem)
    r11 = np.zeros(n_elem)
    r12 = np.zeros(n_elem)
    r21 = np.zeros(n_elem)
    r22 = np.zeros(n_elem)

    betx[0] = betx0
    alfx[0] = alfx0
    bety[0] = bety0
    alfy[0] = alfy0
    r11[0] = RR_ET[0, 0]
    r12[0] = RR_ET[0, 1]
    r21[0] = RR_ET[1, 0]
    r22[0] = RR_ET[1, 1]

    for ii in range(n_elem - 1):

        # Build 2D R matrix of the element
        WW1 = WW[ii, :, :]
        WW2 = WW[ii+1, :, :]
        WW1_inv = lnf.S.T @ WW1.T @ lnf.S
        Rot_e_ii = np.zeros((6,6), dtype=np.float64)
        Rot_e_ii[0:2,0:2] = lnf.Rot2D(2*np.pi*(mux[ii+1] - mux[ii]))
        Rot_e_ii[2:4,2:4] = lnf.Rot2D(2*np.pi*(muy[ii+1] - muy[ii]))
        RRe_ii = WW2 @ Rot_e_ii @ WW1_inv

        # Blocks of the R matrix of the element
        AA = RRe_ii[:2, :2]
        BB = RRe_ii[:2, 2:4]
        CC = RRe_ii[2:4, :2]
        DD = RRe_ii[2:4, 2:4]

        # Propagate EE, FF and RR_ET through the element
        # Bases on MAD-X implementation (see madx/src/twiss.f90, subroutine twcptk)

        if np.allclose(BB, 0, atol=1e-12) and np.allclose(CC, 0, atol=1e-12):
            # Case in which the matrix is block diagonal (no coupling in the element)
            EE = AA
            FF = DD
            EEBAR = SS2D @ EE.T @ SS2D.T
            edet = EE[0,0]*EE[1,1] - EE[0,1]*EE[1,0]
            CCDD = -FF @ RR_ET
            RR_ET = -CCDD @ EEBAR / edet
        else:
            RR_ET_BAR = SS2D @ RR_ET.T @ SS2D.T
            EE = AA - BB @ RR_ET
            edet = EE[0,0]*EE[1,1] - EE[0,1]*EE[1,0]
            EEBAR = SS2D @ EE.T @ SS2D.T
            CCDD = CC - DD @ RR_ET
            FF = DD + CC @ RR_ET_BAR
            RR_ET = -CCDD @ EEBAR / edet

        # Propagate Edwards-Teng Twiss parameters through the element
        # Based on MAD-X implementation (see madx/src/twiss.f90, subroutine twcptk_twiss)

        betx1 = betx[ii]
        alfx1 = alfx[ii]
        bety1 = bety[ii]
        alfy1 = alfy[ii]

        Rx11 = EE[0,0]
        Rx12 = EE[0,1]
        Rx21 = EE[1,0]
        Rx22 = EE[1,1]
        detx = Rx11 * Rx22 - Rx12 * Rx21
        tempb = Rx11 * betx1 - Rx12 * alfx1
        tempa = Rx21 * betx1 - Rx22 * alfx1
        alfx2 = - (tempa * tempb + Rx12 * Rx22) / (detx*betx1)
        betx2 =   (tempb * tempb + Rx12 * Rx12) / (detx*betx1)

        Ry11 = FF[0,0]
        Ry12 = FF[0,1]
        Ry21 = FF[1,0]
        Ry22 = FF[1,1]
        dety = Ry11 * Ry22 - Ry12 * Ry21
        tempb = Ry11 * bety1 - Ry12 * alfy1
        tempa = Ry21 * bety1 - Ry22 * alfy1
        alfy2 = - (tempa * tempb + Ry12 * Ry22) / (dety*bety1)
        bety2 =   (tempb * tempb + Ry12 * Ry12) / (dety*bety1)

        betx[ii+1] = betx2
        alfx[ii+1] = alfx2
        r11[ii+1] = RR_ET[0, 0]
        r12[ii+1] = RR_ET[0, 1]
        r21[ii+1] = RR_ET[1, 0]
        r22[ii+1] = RR_ET[1, 1]
        bety[ii+1] = bety2
        alfy[ii+1] = alfy2

    out_dict = {
        'betx': betx,
        'alfx': alfx,
        'bety': bety,
        'alfy': alfy,
        'r11': r11,
        'r12': r12,
        'r21': r21,
        'r22': r22
    }

    return out_dict