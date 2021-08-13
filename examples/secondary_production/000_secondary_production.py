import numpy as np

import xobjects as xo
import xtrack as xt

context = xo.ContextCpu()

particles = xt.Particles(
        p0c=7000, x=np.linspace(-1e-3, 1e-3, 1000))


class DummyInteractionProcess:

    def __init__(self, fraction_lost, fraction_secondary):

        self.fraction_lost = fraction_lost
        self.fraction_secondary = fraction_secondary

    def interact(self, particles):

        n_part = particles._num_active_particles

        # Kill ~10% of the particles
        mask_kill = np.random.uniform(size=n_part) < self.fraction_lost
        particles.state[:n_part][mask_kill] = 0


        # Generate  
        mask_secondary = np.random.uniform(size=n_part) < self.fraction_secondary
        products['s'] = particles.s[:n_part][mask_secondary]
        products['x'] = particles.x[:n_part][mask_secondary]
        products['px'] = particles.px[:n_part][mask_secondary]
        products['y'] = particles.y[:n_part][mask_secondary]
        products['py'] = particles.py[:n_part][mask_secondary]
        products['zeta'] = particles.zeta[:n_part][mask_secondary]
        products['delta'] = particles.delta[:n_part][mask_secondary]

        products['mass_ratio'] = particles.mass_ratio[:n_part][mask_secondary] * .5
        products['charge_ratio'] = particles.charge_ratio[:n_part][mask_secondary] * .5

        products['parent_id'] = particles.particle_id[:n_part][mask_secondary]

class BeamInteraction:

    def __init__(self, interaction_process):
        self.interaction_process = interaction_process

    def track(self, particles):

        assert isinstance(particles.bufer.context, xo.ContextCpu)
        assert particles._num_active_particles >= 0

        # Assumes active particles are contiguous
        products = self.interaction_process.interact(particles)

        # TODO: This should work also when no products are there
        #       Particles reorganization should still happen

        if products is None or products.x.size == 0:
            particles.reorganize()
        else:
            new_particles = xt.Particles(_context=particles.buffer.context,
                    p0c = particles.p0c[0], # TODO: Should we check that 
                                            #       they are all the same?
                    s = products['s'],
                    x = products['x'],
                    px = products['px'],
                    y = products['y'],
                    py = products['py'],
                    zeta = products['zeta'],
                    delta = products['delta'],
                    mass_ratio = products['mass_ratio'],
                    charge_ratio = products['charge_ratio'],
                    parent_id = products['parent_id'])

            particles.add_particles(new_particles)



