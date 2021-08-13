import xobjects as xo
import xtraxk as xt

context = xo.ContextCpu()

particles = xt.Particles(
        p0c=7000, x=np.linpace(-1e-3, 1e-3, 1000))



class InteractionProcess:
    pass

class BeamInteraction:

    def __init__(self, interaction_process):
        self.interaction_process = interaction_process

    def track(self, particles):

        products = self.interaction_process.interact(particles)

        new_particles = xt.Particles(_context=particles.buffer.context,
                p0c = particles.p0c[0], # TODO: Should we check that 
                                        #       they are all the same?
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



