import xtrack as xt



class EnvParticles:

    def __init__(self):
        self._particles = {}

    def __getitem__(self, key):
        return self._particles[key]

    def __setitem__(self, key, value):
        self._particles[key] = value
        self._particles[key].label = key

    def to_dict(self):
        return {key: particle.to_dict() for key, particle in self._particles.items()}

    @classmethod
    def from_dict(cls, dct):
        instance = cls()
        for key, value in dct.items():
            instance[key] = xt.Particles.from_dict(value)
        return instance


class LineParticleRef:

    def __get__(self, line, objtype=None):
        part = line._extra_config['particle_ref']
        if isinstance(part, str):
            return line.env.particles[part]
        return part

    def __set__(self, line, value):
        line._extra_config['particle_ref'] = value

xt.Line.particle_ref = LineParticleRef()

env = xt.Environment()
env.particles = EnvParticles()

pref = env.ref_manager.ref(env.particles, 'particles')

env._var_management['pref'] = pref

line = env.new_line(name='b1')

line._extra_config['particle_ref'] = None

env.particles['particle/b1'] = xt.Particles(mass0=xt.PROTON_MASS_EV, q0=1)
line.particle_ref = 'particle/b1'







# lhc.particles.new(name='proton', mass0=xt.PROTON_MASS_EV, q0=1,
#                   anomalous_magnetic_moment=1.7928473565, pdg_id=2212)

# lhc.particles.new('particle/b1', 'proton', p0c='nrg * 1e9', line='b1')

# lhc.b2.particle_ref = 'particle_ref/b1'

# lhc.b2.particle_ref['p0c']
# lhc.b2.particle_ref.label # is 'particle_ref/b1'
# lhc.b2.particle_ref['p0c'] = 3.

# lhc['vvv']= lhc.ref['particle/b1'].gamma0[0]


