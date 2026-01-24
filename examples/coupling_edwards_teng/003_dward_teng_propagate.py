import xtrack as xt
import numpy as np
import matplotlib.pyplot as plt
from cpymad.madx import Madx
import xobjects as xo

from rdt_calculation import compute_rdt

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

twmad = mad.twiss()

line = xt.Line.from_madx_sequence(mad.sequence.lhcb1, deferred_expressions=True)
line.particle_ref = xt.Particles(p0c=450e9)
tw = line.twiss4d(coupling_edw_teng=True)


r11_mad, r12_mad, r21_mad, r22_mad = twmad.r11, twmad.r12, twmad.r21, twmad.r22

s_mad = twmad.s

r11_mad_at_s = np.interp(tw.s, s_mad, r11_mad)
r12_mad_at_s = np.interp(tw.s, s_mad, r12_mad)
r21_mad_at_s = np.interp(tw.s, s_mad, r21_mad)
r22_mad_at_s = np.interp(tw.s, s_mad, r22_mad)
betx_mad_at_s = np.interp(tw.s, s_mad, twmad.betx)
bety_mad_at_s = np.interp(tw.s, s_mad, twmad.bety)
alfx_mad_at_s = np.interp(tw.s, s_mad, twmad.alfx)
alfy_mad_at_s = np.interp(tw.s, s_mad, twmad.alfy)

rdt_mad_at_s = compute_rdt(r11_mad_at_s, r12_mad_at_s, r21_mad_at_s, r22_mad_at_s,
                           betx_mad_at_s, bety_mad_at_s, alfx_mad_at_s, alfy_mad_at_s)

# Compute element R matrix
WW = tw.W_matrix
WW_inv = np.linalg.inv(WW)

lnf = xt.linear_normal_form
SS2D = lnf.S[:2, :2]

# Rot_e = WW * 0
# n_elem = len(tw.s)
# lnf = xt.linear_normal_form
# for ii in range(n_elem-1):
#     Rot_e[ii, 0:2, 0:2] = lnf.Rot2D(tw.mux[ii+1] - tw.mux[ii])
#     Rot_e[ii, 2:4, 2:4] = lnf.Rot2D(tw.muy[ii+1] - tw.muy[ii])

RR_ET0 = np.array([[tw.r11_edw_teng[0], tw.r12_edw_teng[0]],
                  [tw.r21_edw_teng[0], tw.r22_edw_teng[0]]])

# Rot = np.zeros(shape=(6, 6), dtype=np.float64)

# Rot[0:2,0:2] = lnf.Rot2D(tw.qx)
# Rot[2:4,2:4] = lnf.Rot2D(tw.qy)

# RR0 = WW[0, :, :] @ Rot @ WW_inv[0, :, :]

# AA0 = RR0[:2, :2]
# BB0 = RR0[:2, 2:4]
# CC0 = RR0[2:4, :2]
# DD0 = RR0[2:4, 2:4]



# tr = np.linalg.trace


# cbar0 = -SS2D @ CC0.T @ SS2D.T
# aux0 = BB0 + cbar0
# det0 = aux0[0,0] * aux0[1,1] - aux0[0,1] * aux0[1,0]

# dtr0 = tr(AA0) - tr(DD0)
# arg0 = dtr0**2 - 4 * np.linalg.det(aux0)


# gammacp0 = sqrt(0.5 + 0.5*sqrt(dtr0**2/arg0))



# gammacp = gammacp0
RR_ET = RR_ET0.copy()

betx = [tw.betx_edw_teng[0]]
alfx = [tw.alfx_edw_teng[0]]
n_elem = len(tw.s)
r11 = [tw.r11_edw_teng[0]]
r12 = [tw.r12_edw_teng[0]]
r21 = [tw.r21_edw_teng[0]]
r22 = [tw.r22_edw_teng[0]]
for ii in range(n_elem - 1):

    # Build R matrix of the element
    WW1 = WW[ii, :, :]
    WW2 = WW[ii+1, :, :]
    Rot_e_ii = np.zeros((6,6), dtype=np.float64)
    Rot_e_ii[0:2,0:2] = xt.linear_normal_form.Rot2D(2*np.pi*(tw.mux[ii+1] - tw.mux[ii]))
    Rot_e_ii[2:4,2:4] = xt.linear_normal_form.Rot2D(2*np.pi*(tw.muy[ii+1] - tw.muy[ii]))
    RRe_ii = WW2 @ Rot_e_ii @ np.linalg.inv(WW1)

    # Blocks of the R matrix of the element
    AA = RRe_ii[:2, :2]
    BB = RRe_ii[:2, 2:4]
    CC = RRe_ii[2:4, :2]
    DD = RRe_ii[2:4, 2:4]

    # Case in which the matrix is block diagonal
    if np.allclose(BB, 0, atol=1e-10) and np.allclose(CC, 0, atol=1e-10):
        EE = AA
        FF = DD
        EEBAR = SS2D @ EE.T @ SS2D.T
        edet = np.linalg.det(EE)
        CCDD = -FF @ RR_ET
        RR_ET = -CCDD @ EEBAR / edet
    else:
        RR_ET_BAR = SS2D @ RR_ET.T @ SS2D.T

        EE = AA - BB @ RR_ET
        edet = np.linalg.det(EE)
        EEBAR = SS2D @ EE.T @ SS2D.T
        CCDD = CC - DD @ RR_ET
        FF = DD + CC @ RR_ET_BAR
        RR_ET = -CCDD @ EEBAR / edet

    r11.append(RR_ET[0, 0])
    r12.append(RR_ET[0, 1])
    r21.append(RR_ET[1, 0])
    r22.append(RR_ET[1, 1])


    betx1 = betx[-1]
    alfx1 = alfx[-1]
    # # RRx_ii = RRe_ii[0:2, 0:2]
    # # RRx_ii = EE
    # r11x = EE[0, 0]
    # r12x = EE[0, 1]
    # r21x = EE[1, 0]
    # r22x = EE[1, 1]
    # det_rx = r11x * r22x - r12x * r21x
    # bet2 = 1/betx1/abs(det_rx) * ((r11x*betx1 - r12x*alfx1)**2 + r12x**2)
    # alfx2 = -1/betx1/abs(det_rx) * ((r11x*betx1 - r12x*alfx1)*(r21x*betx1 - r22x*alfx1) + r12x*r22x)

    matx11 = EE[0,0]
    matx12 = EE[0,1]
    matx21 = EE[1,0]
    matx22 = EE[1,1]

    detx = matx11 * matx22 - matx12 * matx21

    tempb = matx11 * betx1 - matx12 * alfx1
    tempa = matx21 * betx1 - matx22 * alfx1
    alfx2 = - (tempa * tempb + matx12 * matx22) / (detx*betx1)
    betx2 =   (tempb * tempb + matx12 * matx12) / (detx*betx1)

    betx.append(betx2)
    alfx.append(alfx2)

betx = np.array(betx)
alfx = np.array(alfx)