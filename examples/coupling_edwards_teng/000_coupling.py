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

twmad = mad.twiss()

line = xt.Line.from_madx_sequence(mad.sequence.lhcb1, deferred_expressions=True)
line.particle_ref = xt.Particles(p0c=450e9)
tw = line.twiss4d(compute_coupling_elements_ed_teng=True)

ngtw = line.madng_twiss()

r11_mad, r12_mad, r21_mad, r22_mad = twmad.r11, twmad.r12, twmad.r21, twmad.r22
s_mad = twmad.s
r11_ng, r12_ng, r21_ng, r22_ng = ngtw.r11_ng, ngtw.r12_ng, ngtw.r21_ng, ngtw.r22_ng
s_ng = ngtw.s

r11_mad_at_s = np.interp(tw.s, s_mad, r11_mad)
r12_mad_at_s = np.interp(tw.s, s_mad, r12_mad)
r21_mad_at_s = np.interp(tw.s, s_mad, r21_mad)
r22_mad_at_s = np.interp(tw.s, s_mad, r22_mad)

sgn_ng = np.sign(tw.r11 / r11_ng)

# Compare results with MAD-X
plt.close('all')
fig, ((ax0, ax1), (ax2, ax3), (ax4, ax5)) = plt.subplots(3, 2, figsize=(10, 12), sharex=True)
ax0.plot(s_mad, r11_mad, label='MAD-X')
ax0.plot(s_ng, (r11_ng * sgn_ng), label='MAD-NG', linestyle='--')
ax0.plot(tw.s, (tw.r11.real), label='Xsuite', linestyle=':')
ax0.plot(tw.s, np.abs(r11_mad_at_s - tw.r11.real), label='absolute error')
ax0.legend()

ax1.plot(s_mad, r12_mad, label='MAD-X')
ax1.plot(s_ng, (r12_ng * sgn_ng), label='MAD-NG', linestyle='--')
ax1.plot(tw.s, (tw.r12.real), label='Xsuite', linestyle=':')
ax1.plot(tw.s, np.abs(r12_mad_at_s - tw.r12.real), label='absolute error')
ax1.legend()

ax2.plot(s_mad, r21_mad, label='MAD-X')
ax2.plot(s_ng, (r21_ng * sgn_ng), label='MAD-NG', linestyle='--')
ax2.plot(tw.s, (tw.r21.real), label='Xsuite', linestyle=':')
ax2.plot(tw.s, np.abs(r21_mad_at_s - tw.r21.real), label='absolute error')
ax2.legend()

ax3.plot(s_mad, r22_mad, label='MAD-X')
ax3.plot(s_ng, (r22_ng * sgn_ng), label='MAD-NG', linestyle='--')
ax3.plot(tw.s, (tw.r22.real), label='Xsuite', linestyle=':')
ax3.plot(tw.s, np.abs(r22_mad_at_s - tw.r22.real), label='absolute error')
ax3.legend()

ax4.plot(tw.s, tw.f1001, label='Xsuite')
ax4.plot(ngtw.s, sgn_ng * ngtw.f1001_ng, label='MAD-NG', linestyle='--')
ax4.legend()

ax5.plot(tw.s, tw.f1010, label='Xsuite')
ax5.plot(ngtw.s, sgn_ng * ngtw.f1010_ng, label='MAD-NG', linestyle='--')
ax5.legend()


ax0.set_title('r11')
ax1.set_title('r12')
ax2.set_title('r21')
ax3.set_title('r22')
plt.show()

xo.assert_allclose(tw.r11, r11_mad_at_s, atol=1e-3)
xo.assert_allclose(tw.r12, r12_mad_at_s, atol=0.135)
xo.assert_allclose(np.mean(tw.r12 - r12_mad_at_s), 0, atol=2e-5)
xo.assert_allclose(tw.r21, r21_mad_at_s, atol=1e-4)
xo.assert_allclose(tw.r22, r22_mad_at_s, atol=1e-2)
