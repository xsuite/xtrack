# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import time
import numpy as np
from scipy.constants import c as clight

import xtrack as xt

###################
# Load ring model #
###################

env = xt.load('../../test_data/clic_dr/sequence.madx')
line = env.ring
line.set_particle_ref('electron', energy0=2.86e9)
line['rf'].frequency = 2852 / line.get_length() * clight

##############################
# Configure element modeling #
##############################

# Inspect line table
tt = line.get_table()
tt_quad = tt.rows[tt.element_type=='Quadrupole']
tt_bend = tt.rows[tt.element_type=='Bend']
tt_sext = tt.rows[tt.element_type=='Sextupole']
tt_wig = tt.rows['wig.*']

# Set models and integrators
line.set(tt_quad, model='mat-kick-mat', integrator='uniform', num_multipole_kicks=2)
line.set(tt_sext, model='drift-kick-drift-expanded', integrator='uniform', num_multipole_kicks=2)
line.set(tt_bend, model='drift-kick-drift-expanded', integrator='uniform', num_multipole_kicks=4)
line.set(tt_wig, model='drift-kick-drift-expanded', integrator='uniform', num_multipole_kicks=2)

########################
# Slice thick elements #
########################

# Slice thick elements to have more points in twiss table (better radiation integrals)
line.slice_thick_elements(
    slicing_strategies=[
        xt.Strategy(slicing=None), # default, no slicing
        xt.Strategy(slicing=xt.Uniform(4, mode='thick'), element_type=xt.Bend),
        xt.Strategy(slicing=xt.Uniform(2, mode='thick'), element_type=xt.Quadrupole),
        xt.Strategy(slicing=xt.Uniform(2, mode='thick'), element_type=xt.Sextupole),
        xt.Strategy(slicing=xt.Uniform(2, mode='thick'), name='wig.*')
])


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
#  - tw['partition_numbers'] provides the corresponding damping partion numbers.
#  - tw['eq_nemitt_x'] provides the equilibrium horizontal emittance.
#  - tw['eq_nemitt_y'] provides the equilibrium vertical emittance.
#  - tw['eq_nemitt_zeta'] provides the equilibrium longitudinal emittance.

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
line.track(particles, num_turns=num_turns, turn_by_turn_monitor=True,
           with_progress=10)

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
line.track(particles, num_turns=num_turns, turn_by_turn_monitor=True,
           with_progress=10)
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

