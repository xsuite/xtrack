import xtrack as xt

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

line = env.new_line(name='b1')

line._extra_config['particle_ref'] = None

env.particles['particle/b1'] = xt.Particles(mass0=xt.PROTON_MASS_EV, q0=1)
line.particle_ref = 'particle/b1'

env2 = xt.Environment.from_dict(env.to_dict())







# lhc.particles.new(name='proton', mass0=xt.PROTON_MASS_EV, q0=1,
#                   anomalous_magnetic_moment=1.7928473565, pdg_id=2212)

# lhc.particles.new('particle/b1', 'proton', p0c='nrg * 1e9', line='b1')

# lhc.b2.particle_ref = 'particle_ref/b1'

# lhc.b2.particle_ref['p0c']
# lhc.b2.particle_ref.label # is 'particle_ref/b1'
# lhc.b2.particle_ref['p0c'] = 3.

# lhc['vvv']= lhc.ref['particle/b1'].gamma0[0]


