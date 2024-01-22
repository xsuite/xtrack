# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np

import xobjects as xo
import xtrack as xt

context = xo.ContextCpu()

######################################
# Create a dummy collimation process #
######################################

class DummyInteractionProcess:

    def __init__(self, fraction_lost, fraction_secondary):

        self.fraction_lost = fraction_lost
        self.fraction_secondary = fraction_secondary

    def interact(self, particles):

        n_part = particles._num_active_particles

        # Kill some particles
        mask_kill = np.random.uniform(size=n_part) < self.fraction_lost
        particles.state[:n_part][mask_kill] = 0


        # Generate some more particles
        mask_secondary = np.random.uniform(size=n_part) < self.fraction_secondary
        n_products = np.sum(mask_secondary)
        if n_products>0:
            products = {
                's': particles.s[:n_part][mask_secondary],
                'x': particles.x[:n_part][mask_secondary],
                'px': particles.px[:n_part][mask_secondary],
                'y': particles.y[:n_part][mask_secondary],
                'py': particles.py[:n_part][mask_secondary],
                'zeta': particles.zeta[:n_part][mask_secondary],
                'delta': particles.delta[:n_part][mask_secondary],

                'mass_ratio': particles.x[:n_part][mask_secondary] *0 + .5,
                'charge_ratio': particles.x[:n_part][mask_secondary] *0 + .5,

                'parent_particle_id': particles.particle_id[:n_part][mask_secondary],
                'at_element': particles.at_element[:n_part][mask_secondary],
                'at_turn': particles.at_turn[:n_part][mask_secondary],
                }
        else:
            products = None

        return products

#############################################
# Create the corresponding beam interaction #
#############################################

beam_interaction = xt.BeamInteraction(
        interaction_process=DummyInteractionProcess(fraction_lost=0.1,
                                                    fraction_secondary=0.2))

############################################
# Go through the collimator multiple times #
############################################

particles = xt.Particles(_capacity=200,
        p0c=7000, x=np.linspace(-1e-3, 1e-3, 10))

for _ in range(10):
    beam_interaction.track(particles)

###############
# Some checks #
###############

assert particles._num_lost_particles >= 0
assert particles._num_active_particles >= 0

n_all_parts = particles._num_active_particles + particles._num_lost_particles

assert np.all(np.diff(particles.state) <= 0) # checks that there is no lost after active

# Check each id is present only once
ids = particles.particle_id[:n_all_parts]
assert len(list(set(ids))) == n_all_parts

# Check parent and secondaries have the same position
ind_secondaries = np.where(particles.parent_particle_id[:particles._num_active_particles] !=
                    particles.particle_id[:particles._num_active_particles])[0]
for ii in ind_secondaries:
    parent_id = particles.parent_particle_id[ii]
    parent_x = particles.x[np.where(particles.particle_id == parent_id)[0][0]]

    assert parent_x == particles.x[ii]
