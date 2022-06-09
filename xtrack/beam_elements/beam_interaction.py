import numpy as np

import xobjects as xo
import xpart as xp


class BeamInteraction:
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
            new_particles = xp.Particles(_context=particles._buffer.context,
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
