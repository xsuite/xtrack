import json
import numpy as np
import xtrack as xt
import xobjects as xo

from scipy.constants import c as clight

#########################################
# Load line and twiss with no radiation #
#########################################

filename = '../../test_data/clic_dr/line_for_taper.json'

with open(filename, 'r') as f:
    line = xt.Line.from_dict(json.load(f))
line.build_tracker()

line['rf3'].voltage = 0. # desymmetrize the rf
line['rf'].voltage = 0.  # desymmetrize the rf

line['rf1'].voltage *= 2
line['rf2b'].voltage *= 2
line['rf2a'].voltage *= 2

# Use harmonic for one of the cavities
t_rev = line.get_length() / clight
line['rf1'].harmonic = line['rf1'].frequency * t_rev
line['rf1'].frequency = 0

line.particle_ref.p0c = 4e9  # eV

line.configure_radiation(model=None)
tw_no_rad = line.twiss(method='4d')

###############################################
# Enable radiation and compensate energy loss #
###############################################

line.configure_radiation(model='mean')

# - Set cavity lags to compensate energy loss
# - Taper magnet strengths to avoid optics and orbit distortions
line.compensate_radiation_energy_loss(max_iter=100, delta0='zero_mean')

##############################
# Twiss to check the results #
##############################

tw = line.twiss(method='6d')

tw.delta # contains the momentum deviation along the ring

#!end-doc-part

p0corr = 1 + tw.delta

delta_ave = np.trapezoid(tw.delta, tw.s)/tw.s[-1]
xo.assert_allclose(delta_ave, 0, rtol=0, atol=1e-6)

xo.assert_allclose(tw.qx, tw_no_rad.qx, rtol=0, atol=5e-4)
xo.assert_allclose(tw.qy, tw_no_rad.qy, rtol=0, atol=5e-4)

xo.assert_allclose(tw.dqx, tw_no_rad.dqx, rtol=0, atol=1.5e-2*tw.qx)
xo.assert_allclose(tw.dqy, tw_no_rad.dqy, rtol=0, atol=1.5e-2*tw.qy)

xo.assert_allclose(tw.x, tw_no_rad.x, rtol=0, atol=1e-7)
xo.assert_allclose(tw.y, tw_no_rad.y, rtol=0, atol=1e-7)

xo.assert_allclose(tw.betx*p0corr, tw_no_rad.betx, rtol=2e-2, atol=0)
xo.assert_allclose(tw.bety*p0corr, tw_no_rad.bety, rtol=2e-2, atol=0)

xo.assert_allclose(tw.dx, tw.dx, rtol=0.0, atol=0.1e-3)

xo.assert_allclose(tw.dy, tw.dy, rtol=0.0, atol=0.1e-3)

import matplotlib.pyplot as plt
plt.close('all')

fig1 = plt.figure(1, figsize=(6.4, 4.8*1.5))
spbet = plt.subplot(4,1,1)
spco = plt.subplot(4,1,2, sharex=spbet)
spdisp = plt.subplot(4,1,3, sharex=spbet)
spdelta = plt.subplot(4,1,4, sharex=spbet)

spbet.plot(tw['s'], tw['betx'])
spbet.plot(tw['s'], tw['bety'])

spco.plot(tw['s'], tw['x'])
spco.plot(tw['s'], tw['y'])

spdisp.plot(tw['s'], tw['dx'])
spdisp.plot(tw['s'], tw['dy'])

spdelta.plot(tw['s'], tw['delta'])

spbet.set_ylabel(r'$\beta_{x,y}$ [m]')
spco.set_ylabel(r'(Closed orbit)$_{x,y}$ [m]')
spdisp.set_ylabel(r'$D_{x,y}$ [m]')
spdelta.set_ylabel(r'$\Delta P / P_0$')
spdelta.set_xlabel('s [m]')

fig1.suptitle(
    r'$q_x$ = ' f'{tw["qx"]:.5f}' r' $q_y$ = ' f'{tw["qy"]:.5f}' '\n'
    r"$Q'_x$ = " f'{tw["dqx"]:.2f}' r" $Q'_y$ = " f'{tw["dqy"]:.2f}'
    r' $\gamma_{tr}$ = '  f'{1/np.sqrt(tw["momentum_compaction_factor"]):.2f}'
)

tt = line.get_table(attr=True)

fig1.subplots_adjust(left=.15, right=.92, hspace=.27)
plt.show()