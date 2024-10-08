import numpy as np
import xtrack as xt
import xobjects as xo

# Load a line and build tracker
line = xt.Line.from_json(
    '../../test_data/hllhc14_no_errors_with_coupling_knobs/line_b1.json')
line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, q0=1, energy0=7e12)
line.cycle('ip1', inplace=True)
line.twiss_default['method'] = '4d'

# Flat machine
for nn in line.vars.get_table().rows['on_.*|corr_.*'].name:
    line.vars[nn] = 0

line['cmrskew'] = 0
line['cmiskew'] = 0
tw0 = line.twiss()

line['cmrskew'] = 0.5e-4
line['cmiskew'] = -0.3e-4

tw = line.twiss(strengths = True)

c_min_from_k1s = (0+0j) * tw.s
for ii in xt.progress_indicator.progress(range(len(tw.s))):
    c_min_from_k1s[ii] = 1 / (2*np.pi) * np.sum(tw.k1sl * np.sqrt(tw0.betx * tw0.bety)
            * np.exp(1j * 2 * np.pi * ((tw0.mux - tw0.mux[ii]) - (tw0.muy - tw0.muy[ii]))))

xo.assert_allclose(tw.c_minus_re + 1j*tw.c_minus_im, c_min_from_k1s, rtol=5e-2, atol=0)
# Check phi1
xo.assert_allclose(tw.c_minus_re + 1j*tw.c_minus_im,
    tw.c_minus * np.exp(1j * tw.c_phi1), rtol=1e-10, atol=0)
# Check phi2
xo.assert_allclose(tw.c_minus_re + 1j*tw.c_minus_im,
    tw.c_minus * np.exp(1j * (np.pi - tw.c_phi2)), rtol=3e-2, atol=0)
# Check r1
xo.assert_allclose(tw.c_r1, np.sqrt(tw.bety1 / tw.betx1), rtol=1e-5, atol=0)
# Check r2
xo.assert_allclose(tw.c_r2, np.sqrt(tw.betx2 / tw.bety2), rtol=1e-5, atol=0)

# Check c_minus
xo.assert_allclose(np.abs(tw.c_minus_re + 1j*tw.c_minus_im), tw.c_minus, rtol=1e-10, atol=0)
xo.assert_allclose(tw.c_minus_re_0, tw.c_minus_re[0], rtol=1e-10, atol=0)
xo.assert_allclose(tw.c_minus_im_0, tw.c_minus_im[0], rtol=1e-10, atol=0)

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.plot(tw.s, np.abs(tw.c_minus_re + 1j*tw.c_minus_im - c_min_from_k1s)/np.abs(c_min_from_k1s))

plt.show()