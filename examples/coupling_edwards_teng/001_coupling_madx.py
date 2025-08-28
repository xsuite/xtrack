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

idx = tw.rows.indices['ip3'][0]

WW = tw.W_matrix[idx, :, :]

WW_inv = np.linalg.inv(WW)

Rot = np.zeros(shape=(6, 6), dtype=np.float64)
lnf = xt.linear_normal_form

Rot[0:2,0:2] = lnf.Rot2D(tw.qx)
Rot[2:4,2:4] = lnf.Rot2D(tw.qy)

RR = WW @ Rot @ WW_inv

AA = RR[:2, :2]
BB = RR[:2, 2:4]
CC = RR[2:4, :2]
DD = RR[2:4, 2:4]

def conj_mat(mm):
    a = mm[0,0]
    b = mm[0,1]
    c = mm[1,0]
    d = mm[1,1]
    return np.array([[d, -b], [-c, a]])
tr = np.linalg.trace
b_pl_c = BB + conj_mat(CC)
det_bc = np.linalg.det(b_pl_c)
tr_a_m_tr_d = tr(AA) - tr(DD)
coeff = - (0.5 * tr_a_m_tr_d
          + np.sign(det_bc) * np.sqrt(det_bc + 0.25 * tr_a_m_tr_d**2))
R_edw_teng = 1/coeff * b_pl_c