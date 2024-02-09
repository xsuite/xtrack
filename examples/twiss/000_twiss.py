# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np
import xtrack as xt

# Load a line and build tracker
line = xt.Line.from_json(
    '../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')
line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, q0=1, energy0=7e12)
line.vars['vrf400'] = 16
line.build_tracker()


# Twiss
tw = line.twiss()

# Inspect tunes and chromaticities
tw.qx # Horizontal tune
tw.qy # Vertical tune
tw.dqx # Horizontal chromaticity
tw.dqy # Vertical chromaticity

# Plot closed orbit and lattice functions
import matplotlib.pyplot as plt
plt.close('all')

fig1 = plt.figure(1, figsize=(6.4, 4.8*1.5))
spbet = plt.subplot(3,1,1)
spco = plt.subplot(3,1,2, sharex=spbet)
spdisp = plt.subplot(3,1,3, sharex=spbet)

spbet.plot(tw.s, tw.betx)
spbet.plot(tw.s, tw.bety)
spbet.set_ylabel(r'$\beta_{x,y}$ [m]')

spco.plot(tw.s, tw.x)
spco.plot(tw.s, tw.y)
spco.set_ylabel(r'(Closed orbit)$_{x,y}$ [m]')

spdisp.plot(tw.s, tw.dx)
spdisp.plot(tw.s, tw.dy)
spdisp.set_ylabel(r'$D_{x,y}$ [m]')
spdisp.set_xlabel('s [m]')

fig1.suptitle(
    r'$q_x$ = ' f'{tw.qx:.5f}' r' $q_y$ = ' f'{tw.qy:.5f}' '\n'
    r"$Q'_x$ = " f'{tw.dqx:.2f}' r" $Q'_y$ = " f'{tw.dqy:.2f}'
    r' $\gamma_{tr}$ = '  f'{1/np.sqrt(tw.momentum_compaction_factor):.2f}'
)

fig1.subplots_adjust(left=.15, right=.92, hspace=.27)
plt.show()
