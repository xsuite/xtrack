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

# We consider a case in which all RF cavities are off
tab = line.get_table()
tab_cav = tab.rows[tab.element_type == 'Cavity']
for nn in tab_cav.name:
    line[nn].voltage = 0

# For this configuration, `line.twiss()` gives an error because the
# longitudinal motion is not stable.
# In this case, the '4d' method of `line.twiss()` can be used to compute the
# twiss parameters.

tw = line.twiss(method='4d')

#!end-doc-part

import numpy as np

###########################################
# Match a particle distribution (4d mode) #
###########################################

# The '4d' method can also be used to match a particle distribution:

particles = line.build_particles(method='4d',
                    x_norm=[1,2,3,4], # sigmas
                    nemitt_x=2.5e-6, nemitt_y=2.5e-6)

#!end-doc-part

import matplotlib.pyplot as plt

plt.close('all')

fig1 = plt.figure(1, figsize=(6.4, 4.8*1.5))
spbet = plt.subplot(3,1,1)
spco = plt.subplot(3,1,2, sharex=spbet)
spdisp = plt.subplot(3,1,3, sharex=spbet)

spbet.plot(tw['s'], tw['betx'])
spbet.plot(tw['s'], tw['bety'])

spco.plot(tw['s'], tw['x'])
spco.plot(tw['s'], tw['y'])

spdisp.plot(tw['s'], tw['dx'])
spdisp.plot(tw['s'], tw['dy'])

spbet.set_ylabel(r'$\beta_{x,y}$ [m]')
spco.set_ylabel(r'(Closed orbit)$_{x,y}$ [m]')
spdisp.set_ylabel(r'$D_{x,y}$ [m]')
spdisp.set_xlabel('s [m]')

fig1.suptitle(
    r'$q_x$ = ' f'{tw["qx"]:.5f}' r' $q_y$ = ' f'{tw["qy"]:.5f}' '\n'
    r"$Q'_x$ = " f'{tw["dqx"]:.2f}' r" $Q'_y$ = " f'{tw["dqy"]:.2f}'
    r' $\gamma_{tr}$ = '  f'{1/np.sqrt(tw["momentum_compaction_factor"]):.2f}'
)

fig1.subplots_adjust(left=.15, right=.92, hspace=.27)
plt.show()

#!end-doc-part