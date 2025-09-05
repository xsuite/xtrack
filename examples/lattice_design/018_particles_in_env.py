import xtrack as xt

env = xt.Environment()
env['a'] = 4.

env.new_particle('my_particle', "Particles", p0c=['1e12 * a'])
