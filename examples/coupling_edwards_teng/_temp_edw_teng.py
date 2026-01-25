import numpy as np

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