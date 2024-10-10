import numpy as np
from cpymad.madx import Madx

import xtrack as xt

mad = Madx()
mad.call('../../test_data/hllhc15_noerrors_nobb/sequence.madx')
mad.use('lhcb1')

tw_mad_no_coupling = mad.twiss(ripken=True).dframe()

# introduce coupling
mad.sequence.lhcb1.expanded_elements[7].ksl = [0,0.5e-4]

tw_mad_coupling = mad.twiss(ripken=True).dframe()

line = xt.Line.from_madx_sequence(mad.sequence.lhcb1)
line.particle_ref = xt.Particles(p0c=7000e9, mass0=xt.PROTON_MASS_EV)
line.build_tracker()

tw = line.twiss()

bety1 = tw.bety1
betx2 = tw.betx2

twdf = tw.to_pandas()
twdf.set_index('name', inplace=True)

ips = ['ip1', 'ip2', 'ip3', 'ip4', 'ip5', 'ip6', 'ip7', 'ip8']
betx2_at_ips = twdf.loc[ips, 'betx2'].values
bety1_at_ips = twdf.loc[ips, 'bety1'].values

tw_mad_coupling.set_index('name', inplace=True)
beta12_mad_at_ips = tw_mad_coupling.loc[[ip+':1' for ip in ips], 'beta12'].values
beta21_mad_at_ips = tw_mad_coupling.loc[[ip+':1' for ip in ips], 'beta21'].values

assert np.allclose(betx2_at_ips, beta12_mad_at_ips, rtol=1e-4, atol=0)
assert np.allclose(bety1_at_ips, beta21_mad_at_ips, rtol=1e-4, atol=0)

cmin = tw.c_minus

ttmm = mad.twiss(ripken=True)
dqmin_phase_mad = ttmm.summary.dqmin_phase

tt = line.twiss(strengths=True)
WW = tw.W_matrix[0]
# https://www.agsrhichome.bnl.gov//AP/ap_notes/ap_note_174.pdf

k1sl = tt['k1sl']
c_min = 1 / (2*np.pi) * np.sum(k1sl * np.sqrt(tt.betx * tt.bety)
                        * np.exp(1j * 2 * np.pi * (tt.mux - tt.muy)))
# assert np.isclose(cmin, mad.table.summ.dqmin[0], rtol=0, atol=1e-5)

# Eq. 9 in https://www.rhichome.bnl.gov/AP/ap_notes/ap_note_174.pdf
r1 = np.sqrt(WW[2, 0]**2 + WW[2, 1]**2) / WW[0, 0]
r2 = np.sqrt(WW[0, 2]**2 + WW[0, 3]**2) / WW[2, 2]
Delta = np.mod(tt.qx - tt.qy, 1)

# Eq. 10 in https://www.rhichome.bnl.gov/AP/ap_notes/ap_note_174.pdf
phi = np.angle(WW[2, 0] + 1j*WW[2, 1])

WW_ip7 = tw['W_matrix', 'ip7']
phi_ip7 = np.angle(WW_ip7[2, 0] + 1j*WW_ip7[2, 1])
tt_ip7 = line.twiss(zero_at='ip7', strengths=True)
cmin_ip7 = 1 / (2*np.pi) * np.sum(tt_ip7.k1sl * np.sqrt(tt_ip7.betx * tt_ip7.bety)
                        * np.exp(1j * 2 * np.pi * (tt_ip7.mux - tt_ip7.muy)))

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
sp1 = plt.subplot(211)
plt.plot(tw.s, tw.bety1, label='Xsuite')
plt.plot(tw_mad_coupling.s, tw_mad_coupling.beta21, '--', label='Madx')
plt.ylabel(r'$\beta_{1,y}$')
plt.legend(loc='upper right')
plt.subplot(212, sharex=sp1)
plt.plot(tw.s, tw.betx2, label='betx2')
plt.plot(tw_mad_coupling.s, tw_mad_coupling.beta12, '--')
plt.ylabel(r'$\beta_{2,x}$')
plt.suptitle(r'Xsuite: $C^{-}$'
             f" = {cmin:.2e} "
             r"MAD-X: $C^{-}$ = "
             f"{mad.table.summ.dqmin[0]:.2e}"
             )


plt.show()