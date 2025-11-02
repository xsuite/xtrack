# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import xtrack as xt

# Load a line
line = xt.load('../../test_data/hllhc15_noerrors_nobb/line_and_particle.json')
line.set_particle_ref('proton', energy0=7e12)

# Periodic twiss of the full ring
tw_p = line.twiss()

# Periodic twiss of an arc cell
tw = line.twiss(method='4d', start='mq.14r6.b1', end='mq.16r6.b1', init='periodic')

#!end-doc-part

# Plot

# Choose the twiss to plot

import matplotlib.pyplot as plt
plt.close('all')

fig1 = plt.figure(1)
spbet = plt.subplot(2,1,1)
spdisp = plt.subplot(2,1,2, sharex=spbet)

spbet.plot(tw.s, tw.betx, label='x')
spbet.plot(tw.s, tw.bety, label='y')
spbet.set_ylabel(r'$\beta_{x,y}$ [m]')
spbet.set_ylim(bottom=0)
spbet.legend(loc='best')

spdisp.plot(tw.s, tw.dx, label='x')
spdisp.plot(tw.s, tw.dy, label='y')
spdisp.set_ylabel(r'$D_{x,y}$ [m]')
spdisp.set_xlabel('s [m]')

fig1.subplots_adjust(left=.15, right=.92, hspace=.27)
plt.show()