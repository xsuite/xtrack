import xtrack as xt
import xobjects as xo

env = xt.Environment()
env['a'] = 4.

env.new_particle('my_particle', "Particles", p0c=['1e12 * a'])
assert 'my_particle' in env.particles
xo.assert_allclose(env['my_particle'].p0c, 4e12, rtol=0, atol=1e-9)
env['a'] = 5.
xo.assert_allclose(env['my_particle'].p0c, 5e12, rtol=0, atol=1e-9)

env.particle_ref = 'my_particle'
env._particle_ref = 'my_particle'

xo.assert_allclose(env.particle_ref.p0c, 5e12, rtol=0, atol=1e-9)
assert env.particle_ref.__class__.__name__ == 'EnvParticleRef'
env.particle_ref.p0c = '2e12 * a'
xo.assert_allclose(env.particle_ref.p0c, 10e12, rtol=0, atol=1e-9)
env['my_particle'].p0c = '1e12 * a'
xo.assert_allclose(env.particle_ref.p0c, 5e12, rtol=0, atol=1e-9)


env2 = xt.Environment.from_dict(env.to_dict())
assert 'my_particle' in env2.particles
assert env2._particle_ref == "my_particle"
xo.assert_allclose(env2['my_particle'].p0c, 5e12, rtol=0, atol=1e-9)
env2['a'] = 6.
xo.assert_allclose(env2['my_particle'].p0c, 6e12, rtol=0, atol=1e-9)
env2['a'] = 5.

assert env2.particle_ref.__class__.__name__ == 'EnvParticleRef'
env2.particle_ref.p0c = '2e12 * a'
xo.assert_allclose(env2.particle_ref.p0c, 10e12, rtol=0, atol=1e-9)
env2['my_particle'].p0c = '1e12 * a'
xo.assert_allclose(env2.particle_ref.p0c, 5e12, rtol=0, atol=1e-9)

env2 = env.copy()
assert 'my_particle' in env2.particles
xo.assert_allclose(env2['my_particle'].p0c, 5e12, rtol=0, atol=1e-9)
env2['a'] = 6.
xo.assert_allclose(env2['my_particle'].p0c, 6e12, rtol=0, atol=1e-9)
env2['a'] = 5.

assert env2.particle_ref.__class__.__name__ == 'EnvParticleRef'
env2.particle_ref.p0c = '2e12 * a'
xo.assert_allclose(env2.particle_ref.p0c, 10e12, rtol=0, atol=1e-9)
env2['my_particle'].p0c = '1e12 * a'
xo.assert_allclose(env2.particle_ref.p0c, 5e12, rtol=0, atol=1e-9)

ll = env.new_line(name='my_line', components=[])
assert ll._particle_ref == 'my_particle'

xo.assert_allclose(ll.particle_ref.p0c, 5e12, rtol=0, atol=1e-9)
assert ll.particle_ref.__class__.__name__ == 'LineParticleRef'
ll.particle_ref.p0c = '2e12 * a'
xo.assert_allclose(env.particle_ref.p0c, 10e12, rtol=0, atol=1e-9)
env['my_particle'].p0c = '1e12 * a'
xo.assert_allclose(ll.particle_ref.p0c, 5e12, rtol=0, atol=1e-9)

ll2 = xt.Line.from_dict(ll.to_dict())
assert 'my_particle' in ll2.env.particles
assert ll2.env.particle_ref is None
assert ll2.particle_ref.__class__.__name__ == 'LineParticleRef'
assert ll2._particle_ref == 'my_particle'
xo.assert_allclose(ll2.env['my_particle'].p0c, 5e12, rtol=0, atol=1e-9)
ll2['a'] = 7.
xo.assert_allclose(ll2.env['my_particle'].p0c, 7e12, rtol=0, atol=1e-9)
ll2['a'] = 5.

ll2.particle_ref.p0c = '2e12 * a'
xo.assert_allclose(ll2.particle_ref.p0c, 10e12, rtol=0, atol=1e-9)
ll2.env['my_particle'].p0c = '1e12 * a'
xo.assert_allclose(ll2.particle_ref.p0c, 5e12, rtol=0, atol=1e-9)


ll2 = ll.copy()
assert 'my_particle' in ll2.env.particles
assert ll2.env.particle_ref is None
assert ll2.particle_ref.__class__.__name__ == 'LineParticleRef'
assert ll2._particle_ref == 'my_particle'
xo.assert_allclose(ll2.env['my_particle'].p0c, 5e12, rtol=0, atol=1e-9)
ll2['a'] = 7.
xo.assert_allclose(ll2.env['my_particle'].p0c, 7e12, rtol=0, atol=1e-9)
ll2['a'] = 5.

ll2.particle_ref.p0c = '2e12 * a'
xo.assert_allclose(ll2.particle_ref.p0c, 10e12, rtol=0, atol=1e-9)
ll2.env['my_particle'].p0c = '1e12 * a'
xo.assert_allclose(ll2.particle_ref.p0c, 5e12, rtol=0, atol=1e-9)