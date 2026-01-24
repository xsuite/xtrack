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

# Rot_e = WW * 0
# n_elem = len(tw.s)
# lnf = xt.linear_normal_form
# for ii in range(n_elem-1):
#     Rot_e[ii, 0:2, 0:2] = lnf.Rot2D(tw.mux[ii+1] - tw.mux[ii])
#     Rot_e[ii, 2:4, 2:4] = lnf.Rot2D(tw.muy[ii+1] - tw.muy[ii])

betx = [tw.betx[0]]
alfx = [tw.alfx[0]]
n_elem = len(tw.s)
for ii in range(n_elem - 1):
    WW1 = WW[ii, :, :]
    WW2 = WW[ii+1, :, :]
    Rot_e_ii = np.zeros((6,6), dtype=np.float64)
    Rot_e_ii[0:2,0:2] = xt.linear_normal_form.Rot2D(tw.mux[ii+1] - tw.mux[ii])
    Rot_e_ii[2:4,2:4] = xt.linear_normal_form.Rot2D(tw.muy[ii+1] - tw.muy[ii])
    RRe_ii = WW2 @ Rot_e_ii @ np.linalg.inv(WW1)

    bet1 = betx[-1]
    alfx1 = alfx[-1]
    RRx_ii = RRe_ii[0:2, 0:2]
    r11x = RRx_ii[0, 0]
    r12x = RRx_ii[0, 1]
    r21x = RRx_ii[1, 0]
    r22x = RRx_ii[1, 1]
    bet2 = 1/bet1 * ((r11x*bet1 - r12x*alfx1)**2 + r12x**2)
    alfx2 = -1/bet1 * ((r11x*bet1 - r12x*alfx1)*(r21x*bet1 - r22x*alfx1) + r12x*r22x)

    betx.append(bet2)
    alfx.append(alfx2)





# Compare results with MAD-X
plt.close('all')
fig, ((ax0, ax1), (ax2, ax3), (ax4, ax5)) = plt.subplots(3, 2, figsize=(6.4*1.3, 4.8*1.8), sharex=True)
ax0.plot(s_mad, r11_mad, label='MAD-X')
ax0.plot(tw.s, tw.r11_edw_teng, label='Xsuite', linestyle=':')
ax0.plot(tw.s, np.abs(r11_mad_at_s - tw.r11_edw_teng), label='absolute error')
ax0.legend()

ax1.plot(s_mad, r12_mad, label='MAD-X')
ax1.plot(tw.s, tw.r12_edw_teng, label='Xsuite', linestyle=':')
ax1.plot(tw.s, np.abs(r12_mad_at_s - tw.r12_edw_teng), label='absolute error')
ax1.legend()

ax2.plot(s_mad, r21_mad, label='MAD-X')
ax2.plot(tw.s, tw.r21_edw_teng, label='Xsuite', linestyle=':')
ax2.plot(tw.s, np.abs(r21_mad_at_s - tw.r21_edw_teng), label='absolute error')
ax2.legend()

ax3.plot(s_mad, r22_mad, label='MAD-X')
ax3.plot(tw.s, tw.r22_edw_teng, label='Xsuite', linestyle=':')
ax3.plot(tw.s, np.abs(r22_mad_at_s - tw.r22_edw_teng), label='absolute error')
ax3.legend()

ax4.plot(tw.s, tw.f1001.real, label='Xsuite')
ax4.plot(tw.s, rdt_mad_at_s['f1001'].real, label='MAD-X', linestyle=':')
ax4.legend()

ax5.plot(tw.s, tw.f1010.real, label='Xsuite')
ax5.plot(tw.s, rdt_mad_at_s['f1010'].real, label='MAD-X', linestyle=':')
ax5.legend()


ax0.set_title('r11')
ax1.set_title('r12')
ax2.set_title('r21')
ax3.set_title('r22')
plt.show()

xo.assert_allclose(tw.r11_edw_teng, r11_mad_at_s,
                   rtol=1e-5, atol=1e-5*np.max(np.abs(r11_mad_at_s)))
xo.assert_allclose(tw.r12_edw_teng, r12_mad_at_s,
                   rtol=1e-5, atol=1e-5*np.max(np.abs(r12_mad_at_s)))
xo.assert_allclose(tw.r21_edw_teng, r21_mad_at_s,
                   rtol=1e-5, atol=1e-5*np.max(np.abs(r21_mad_at_s)))
xo.assert_allclose(tw.r22_edw_teng, r22_mad_at_s,
                   rtol=1e-5, atol=1e-5*np.max(np.abs(r22_mad_at_s)))
xo.assert_allclose(tw.betx_edw_teng, betx_mad_at_s, atol=0, rtol=1e-5)

xo.assert_allclose(tw.betx_edw_teng, betx_mad_at_s, atol=0, rtol=5e-8)
xo.assert_allclose(tw.alfx_edw_teng, alfx_mad_at_s, atol=1e-4, rtol=1e-8)
xo.assert_allclose(tw.bety_edw_teng, bety_mad_at_s, atol=0, rtol=5e-8)
xo.assert_allclose(tw.alfy_edw_teng, alfy_mad_at_s, atol=1e-4, rtol=1e-8)

xo.assert_allclose(tw.f1001, rdt_mad_at_s['f1001'],
                   atol=1e-5 * np.max(np.abs(rdt_mad_at_s['f1001'])))
xo.assert_allclose(tw.f1010, rdt_mad_at_s['f1010'],
                   atol=1e-5 * np.max(np.abs(rdt_mad_at_s['f1010'])))
