import time

import numpy as np

import xobjects as xo
import xtrack as xt
import xline as xl

context = xo.ContextCpu()

######################################
# Create a dummy collimation process #
######################################


class DummyInteractionProcess:
    '''
    I kill some particles and I kick some others by an given angle
    and I generate some secondaries with the opposite angles.
    '''
    def __init__(self, fraction_lost, fraction_secondary, length, kick_x):

        self.fraction_lost = fraction_lost
        self.fraction_secondary = fraction_secondary
        self.kick_x = kick_x
        self.length = length

        self.drift= xt.Drift(length=self.length)


    def interact(self, particles):

        self.drift.track(particles)

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
                'px': particles.px[:n_part][mask_secondary] + self.kick_x,
                'y': particles.y[:n_part][mask_secondary],
                'py': particles.py[:n_part][mask_secondary],
                'zeta': particles.zeta[:n_part][mask_secondary],
                'delta': particles.delta[:n_part][mask_secondary],

                'mass_ratio': particles.x[:n_part][mask_secondary] *0 + 1.,
                'charge_ratio': particles.x[:n_part][mask_secondary] *0 + 1.,

                'parent_particle_id': particles.particle_id[:n_part][mask_secondary],
                }
        else:
            products = None

        return products

#############################################
# Create the corresponding beam interaction #
#############################################

interaction_process=DummyInteractionProcess(length=1., kick_x=4e-3,
                                            fraction_lost=0.0,
                                            fraction_secondary=0.2)
beam_interaction = xt.BeamInteraction(length=interaction_process.length,
                                      interaction_process=interaction_process)

line = xl.Line(elements=[
    xl.Multipole(knl=[0,0]),
    xl.LimitEllipse(a=2e-2, b=2e-2),
    xl.Drift(length=1.),
    xl.Multipole(knl=[0,0]),
    xl.LimitEllipse(a=2e-2, b=2e-2),
    xl.Drift(length=2.),
    beam_interaction,
    xl.Multipole(knl=[0,0]),
    xl.LimitEllipse(a=2e-2, b=2e-2),
    xl.Drift(length=10.),
    xl.LimitEllipse(a=2e-2, b=2e-2),
    xl.Drift(length=10.),
    ])

tracker = xt.Tracker(sequence=line)

particles = xt.Particles(
        _capacity=200000,
        x=np.zeros(100000))

t1 = time.time()
tracker.track(particles)
t2 = time.time()

print(f'{t2-t1=:.2f}')
