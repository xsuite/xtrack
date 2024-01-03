# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import xtrack as xt

# Load a line and build tracker
line = xt.Line.from_json(
    '../../test_data/hllhc15_noerrors_nobb/line_and_particle.json')
line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, q0=1, energy0=7e12)
line.build_tracker()

# Periodic twiss
tw_p = line.twiss()

# Twiss over a range with user-defined initial conditions (at start)
tw1 = line.twiss(start='ip5', end='mb.c24r5.b1',
                betx=0.15, bety=0.15, py=1e-6)


# Twiss over a range with user-defined initial conditions at end
tw2 = line.twiss(start='ip5', end='mb.c24r5.b1', init_at=xt.END,
                alfx=3.50482, betx=131.189, alfy=-0.677173, bety=40.7318,
                dx=1.22515, dpx=-0.0169647)

# Twiss over a range with user-defined initial conditions at arbitrary location
tw3 = line.twiss(start='ip5', end='mb.c24r5.b1', init_at='mb.c14r5.b1',
                 alfx=-0.437695, betx=31.8512, alfy=-6.73282, bety=450.454,
                 dx=1.22606, dpx=-0.0169647)

# Initial conditions can also be taken from an existing twiss table
tw4 = line.twiss(start='ip5', end='mb.c24r5.b1', init_at='mb.c14r5.b1',
                 init=tw_p)

# More explicitly, a `TwissInit` object can be extracted from the twiss table
# and used as initial conditions
tw_init = tw_p.get_twiss_init('mb.c14r5.b1',)
tw5 = line.twiss(start='ip5', end='mb.c24r5.b1', init=tw_init)


#!end-doc-part

# Plot

# Choose the twiss to plot
tw = tw5

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

for nn in ['ip5', 'mb.c14r5.b1', 'mb.c24r5.b1']:
    for ax in [spbet, spco, spdisp]:
        ax.axvline(tw_p['s', nn], color='k', ls='--', alpha=.5)
    spbet.text(tw_p['s', nn], 22000, nn, rotation=90,
        horizontalalignment='right', verticalalignment='top', alpha=.5)

fig1.subplots_adjust(left=.15, right=.92, hspace=.27)
plt.show()