# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np
import xobjects as xo

import xtrack as xt


class BeamInteraction:

    skip_in_loss_location_refinement = True

    def __init__(self, name=None, interaction_process=None,
                 length=0, isthick=None):

        self.name = name
        self.interaction_process = interaction_process

        self.length = length

        if isthick is None:
            isthick = True if length > 0 else False
        self.isthick = isthick

    def track(self, particles):

        assert isinstance(particles._buffer.context, xo.ContextCpu)
        assert particles._num_active_particles >= 0

        # Assumes active particles are contiguous
        products = self.interaction_process.interact(particles)

        # TODO: This should work also when no products are there
        #       Particles reorganization should still happen

        if products is None or products['x'].size == 0:
            particles.reorganize()
        else:
            new_particles = xt.Particles(_context=particles._buffer.context,
                    p0c = particles.p0c[0], # TODO: Should we check that 
                                            #       they are all the same?
                    mass0 = particles.mass0,
                    q0 = particles.q0,
                    s = products['s'],
                    x = products['x'],
                    px = products['px'],
                    y = products['y'],
                    py = products['py'],
                    zeta = products['zeta'],
                    delta = products['delta'],
                    mass_ratio = products['mass_ratio'],
                    charge_ratio = products['charge_ratio'],
                    at_element = products['at_element'],
                    at_turn = products['at_turn'],
                    parent_particle_id = products['parent_particle_id'])

            particles.add_particles(new_particles)

class ParticlesInjectionSample:

    def __init__(self, particles_to_inject, line, element_name, num_particles_to_inject):
        self.particles_to_inject = particles_to_inject.copy()
        self.line = line
        self.element_name = element_name
        self.num_particles_to_inject = num_particles_to_inject

    def track(self, particles):

        if not isinstance(particles._context, xo.ContextCpu):
            raise ValueError('This element only works with CPU context')

        self.particles_to_inject.at_turn += 1

        if not self.num_particles_to_inject:
            return

        assert self.element_name in self.line.element_names
        assert self.line.element_dict[self.element_name] is self
        at_element = self.line.element_names.index(self.element_name)
        s_element = self.line.tracker._tracker_data_base.element_s_locations[at_element]

        # Get random particles to inject
        idx_inject = np.random.default_rng().choice(len(self.particles_to_inject.x),
                                size=self.num_particles_to_inject, replace=False)

        mask_inject = np.zeros(len(self.particles_to_inject.x), dtype=bool)
        mask_inject[idx_inject] = True

        p_inj = self.particles_to_inject.filter(mask_inject)
        p_inj.s = s_element
        p_inj.at_element = at_element
        p_inj.update_p0c_and_energy_deviations(p0c=self.line.particle_ref.p0c[0])

        particles.add_particles(p_inj)