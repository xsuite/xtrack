import xtrack as xt
import numpy as np
import matplotlib.pyplot as plt
from cpymad.madx import Madx
import xobjects as xo

mad = Madx()
mad.call('../../test_data/lhc_2024/lhc.seq')
mad.call('../../test_data/lhc_2024/injection_optics.madx')

mad.beam()
mad.use('lhcb1')

mad.globals.on_x1 = 0
mad.globals.on_x2h = 0
mad.globals.on_x2v = 0
mad.globals.on_x5 = 0
mad.globals.on_x8h = 0
mad.globals.on_x8v = 0

mad.globals.on_sep1 = 0
mad.globals.on_sep2h = 0
mad.globals.on_sep2v = 0
mad.globals.on_sep5 = 0
mad.globals.on_sep8h = 0
mad.globals.on_sep8v = 0

mad.globals.on_a2 = 0
mad.globals.on_a8 = 0

mad.globals['kqs.a67b1'] = 1e-4

twmad = xt.Table(mad.twiss(), _copy_cols=True)

line = xt.Line.from_madx_sequence(mad.sequence.lhcb1, deferred_expressions=True)
line.particle_ref = xt.Particles(p0c=450e9)
tw = line.twiss4d(coupling_edw_teng=True)

def conj_mat(mm):
    a = mm[0,0]
    b = mm[0,1]
    c = mm[1,0]
    d = mm[1,1]
    return np.array([[d, -b], [-c, a]])

Rot = np.zeros(shape=(6, 6), dtype=np.float64)
lnf = xt.linear_normal_form

Rot[0:2,0:2] = lnf.Rot2D(tw.qx)
Rot[2:4,2:4] = lnf.Rot2D(tw.qy)

num_places = tw.W_matrix.shape[0]
r11_edw_teng = np.zeros(num_places)
r12_edw_teng = np.zeros(num_places)
r21_edw_teng = np.zeros(num_places)
r22_edw_teng = np.zeros(num_places)
betx_edw_teng = np.zeros(num_places)
bety_edw_teng = np.zeros(num_places)
alfx_edw_teng = np.zeros(num_places)
alfy_edw_teng = np.zeros(num_places)

idx = tw.rows.indices['ip3'][0]

if True:
# for idx in range(num_places):

    print(f"Place {idx}/{num_places}", end='\r', flush=True)

    WW = tw.W_matrix[idx, :, :]

    WW_inv = np.linalg.inv(WW)

    RR = WW @ Rot @ WW_inv

    AA = RR[:2, :2]
    BB = RR[:2, 2:4]
    CC = RR[2:4, :2]
    DD = RR[2:4, 2:4]

    tr = np.linalg.trace
    b_pl_c = CC + conj_mat(BB)
    det_bc = np.linalg.det(b_pl_c)
    tr_a_m_tr_d = tr(AA) - tr(DD)
    coeff = - (0.5 * tr_a_m_tr_d
            + np.sign(det_bc) * np.sqrt(det_bc + 0.25 * tr_a_m_tr_d**2))
    R_edw_teng = 1/coeff * b_pl_c

    r11_edw_teng[idx] = R_edw_teng[0,0]
    r12_edw_teng[idx] = R_edw_teng[0,1]
    r21_edw_teng[idx] = R_edw_teng[1,0]
    r22_edw_teng[idx] = R_edw_teng[1,1]

    EE = AA - BB@R_edw_teng
    FF = DD + R_edw_teng@BB

    quarter = 0.25
    two = 2.0

    sinmu2 = -EE[0,1]*EE[1,0] - quarter*(EE[0,0] - EE[1,1])**2
    sinmux = np.sign(EE[0,1]) * np.sqrt(abs(sinmu2))
    betx_et = EE[0,1] / sinmux
    alfx_et = (EE[0,0] - EE[1,1]) / (two * sinmux)

    sinmu2 = -FF[0,1]*FF[1,0] - quarter*(FF[0,0] - FF[1,1])**2
    sinmuy = np.sign(FF[0,1]) * np.sqrt(abs(sinmu2))
    bety_et = FF[0,1] / sinmuy
    alfy_et = (FF[0,0] - FF[1,1]) / (two * sinmuy)

