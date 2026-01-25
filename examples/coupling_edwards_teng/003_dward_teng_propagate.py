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

# Extract data from the twiss
WW = tw.W_matrix
mux = tw.mux
muy = tw.muy
qx = tw.qx
qy = tw.qy

lnf = xt.linear_normal_form

# Initial conditions for Edwards-Teng propagation
# Based on MAD-X implementation (see madx/src/twiss.f90, subroutine twcpin)

Rot = np.zeros(shape=(6, 6), dtype=np.float64)
lnf = xt.linear_normal_form

Rot[0:2,0:2] = lnf.Rot2D(2 * np.pi * qx)
Rot[2:4,2:4] = lnf.Rot2D(2 * np.pi * qy)

WW0 = WW[0, :, :]
WW0_inv = lnf.S.T @ WW0.T @ lnf.S
RR = WW0 @ Rot @ WW0_inv

import _temp_edw_teng as temp_et

edw_teng_init = temp_et._compute_edwards_teng_initial(RR)
edw_teng_cols = temp_et._propagate_edwards_teng(
    WW=WW, mux=mux, muy=muy,
    RR_ET0=edw_teng_init['RR_ET0'],
    betx0=edw_teng_init['betx0'],
    alfx0=edw_teng_init['alfx0'],
    bety0=edw_teng_init['bety0'],
    alfy0=edw_teng_init['alfy0']
)