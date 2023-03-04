# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import time
import numpy as np

from cpymad.madx import Madx

import xtrack as xt
import xpart as xp
import xobjects as xo

# Import a thick sequence
mad = Madx()
mad.call('../../test_data/clic_dr/sequence.madx')
mad.use('ring')

# Makethin
mad.input(f'''
select, flag=MAKETHIN, SLICE=4, thick=false;
select, flag=MAKETHIN, pattern=wig, slice=1;
MAKETHIN, SEQUENCE=ring, MAKEDIPEDGE=true;
use, sequence=RING;
''')
mad.use('ring')

# Build xtrack line
print('Build xtrack line...')
line = xt.Line.from_madx_sequence(mad.sequence['RING'])
line.particle_ref = xp.Particles(
        mass0=xp.ELECTRON_MASS_EV,
        q0=-1,
        gamma0=mad.sequence.ring.beam.gamma)

# Build tracker
print('Build tracker ...')
line.build_tracker()

################################
# Enable synchrotron radiation #
################################

# we choose the `mean` mode in which the mean power loss is applied without
# stochastic fluctuations (quantum excitation).
line.configure_radiation(model='mean')

#########
# Twiss #
#########

tw = line.twiss(eneloss_and_damping=True)

# By setting `eneloss_and_damping=True` we can get additional information
# from the twiss for example:
#  - tw['eneloss_turn'] provides the energy loss per turn (in eV).
#  - tw['damping_constants_s'] provides the damping constants in x, y and zeta.
#  - tw['partition_numbers'] provided the corresponding damping partion numbers.

############################################
# Generate particles and track (mean mode) #
############################################

# Build three particles (with action in x,y and zeta respectively)
part_co = tw['particle_on_co']
particles = line.build_particles(
    x_norm=[500., 0, 0], y_norm=[0, 500, 0], zeta=part_co.zeta[0],
    delta=np.array([0,0,1e-2]) + part_co.delta[0],
    nemitt_x=1e-9, nemitt_y=1e-9)

# Save initial state
particles_0 = particles.copy()

# Track
num_turns = 5000
line.track(particles, num_turns=num_turns, turn_by_turn_monitor=True)

# Save monitor
mon_mean_mode = line.record_last_track

############################
# Switch to `quantum` mode #
############################

# We switch to the `quantum` mode in which the power loss from radiation is
# applied including stochastic fluctuations (quantum excitation).
# IMPORTANT: Note that this mode should not be used to compute twiss parameters
#            nor to match particle distributions. For this reason we switch
#            to quantum mode only after having generated the particles.


line.configure_radiation(model='quantum')

# We reuse the initial state saved before
particles = particles_0.copy()

num_turns = 5000
line.track(particles, num_turns=num_turns, turn_by_turn_monitor=True)
mon_quantum_mode = line.record_last_track

import matplotlib.pyplot as plt
plt.close('all')
figs = []
for ii, mon in enumerate([mon_mean_mode, mon_quantum_mode]):
    fig = plt.figure(ii + 1)
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312, sharex=ax1)
    ax3 = fig.add_subplot(313, sharex=ax1)

    ax1.plot(1e3*mon.x[0, :].T)
    ax2.plot(1e3*mon.y[1, :].T)
    ax3.plot(1e3*mon.delta[2, :].T)
    i_turn = np.arange(num_turns)
    ax1.plot(1e3*(part_co.x[0]
        +(mon.x[0,0]-part_co.x[0])*np.exp(-i_turn*tw['damping_constants_turns'][0])))
    ax2.plot(1e3*(part_co.y[0]
        +(mon.y[1,0]-part_co.y[0])*np.exp(-i_turn*tw['damping_constants_turns'][1])))
    ax3.plot(1e3*(part_co.delta[0]
        +(mon.delta[2,0]-part_co.delta[0])*np.exp(-i_turn*tw['damping_constants_turns'][2])))

    ax1.set_ylabel('x [mm]')
    ax2.set_ylabel('y [mm]')
    ax3.set_ylabel('delta [-]')
    ax3.set_xlabel('Turn')

plt.show()

