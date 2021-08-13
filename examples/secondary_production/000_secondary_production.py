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


                'parent_id': particles.particle_id[:n_part][mask_secondary],
                }
        else:
            products = None

        return products

beam_interaction = xt.BeamInteraction(
        interaction_process=DummyInteractionProcess(fraction_lost=0.1,
                                                    fraction_secondary=0.2))

particles = xt.Particles(_capacity=20,
        p0c=7000, x=np.linspace(-1e-3, 1e-3, 10))

beam_interaction.track(particles)
