import xtrack as xt
import numpy as np
import matplotlib.pyplot as plt
from cpymad.madx import Madx
import xobjects as xo

from rdt_calculation import compute_rdt

# TODO: remove inversions and use symplectic properties

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

# RR_ET0 = np.array([[tw.r11_edw_teng[0], tw.r12_edw_teng[0]],
#                   [tw.r21_edw_teng[0], tw.r22_edw_teng[0]]])

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




# Starting point

Rot = np.zeros(shape=(6, 6), dtype=np.float64)
lnf = xt.linear_normal_form

Rot[0:2,0:2] = lnf.Rot2D(2 * np.pi * tw.qx)
Rot[2:4,2:4] = lnf.Rot2D(2 * np.pi * tw.qy)

WW0 = tw.W_matrix[0, :, :]
WW0_inv = lnf.S.T @ WW0.T @ lnf.S
RR = WW0 @ Rot @ WW0_inv

import _temp_edw_teng as temp_et

edw_teng_init = temp_et._compute_edwards_teng_initial(RR)

RR_ET0 = edw_teng_init['RR_ET0']
betx0 = edw_teng_init['betx0']
alfx0 = edw_teng_init['alfx0']
bety0 = edw_teng_init['bety0']
alfy0 = edw_teng_init['alfy0']


RR_ET = RR_ET0.copy()

n_elem = len(tw.s)
betx = np.zeros(n_elem)
alfx = np.zeros(n_elem)
r11 = np.zeros(n_elem)
r12 = np.zeros(n_elem)
r21 = np.zeros(n_elem)
r22 = np.zeros(n_elem)

betx[0] = betx0
alfx[0] = alfx0
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
    Rot_e_ii[0:2,0:2] = lnf.Rot2D(2*np.pi*(tw.mux[ii+1] - tw.mux[ii]))
    Rot_e_ii[2:4,2:4] = lnf.Rot2D(2*np.pi*(tw.muy[ii+1] - tw.muy[ii]))
    RRe_ii = WW2 @ Rot_e_ii @ WW1_inv

    # Blocks of the R matrix of the element
    AA = RRe_ii[:2, :2]
    BB = RRe_ii[:2, 2:4]
    CC = RRe_ii[2:4, :2]
    DD = RRe_ii[2:4, 2:4]

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

    betx1 = betx[ii]
    alfx1 = alfx[ii]

    matx11 = EE[0,0]
    matx12 = EE[0,1]
    matx21 = EE[1,0]
    matx22 = EE[1,1]
    detx = matx11 * matx22 - matx12 * matx21
    tempb = matx11 * betx1 - matx12 * alfx1
    tempa = matx21 * betx1 - matx22 * alfx1
    alfx2 = - (tempa * tempb + matx12 * matx22) / (detx*betx1)
    betx2 =   (tempb * tempb + matx12 * matx12) / (detx*betx1)

    betx[ii+1] = betx2
    alfx[ii+1] = alfx2
    r11[ii+1] = RR_ET[0, 0]
    r12[ii+1] = RR_ET[0, 1]
    r21[ii+1] = RR_ET[1, 0]
    r22[ii+1] = RR_ET[1, 1]