import xtrack as xt

env = xt.Environment()
env['a'] = 1.

env.new('qd', 'Quadrupole', length=1, k1='-0.05 * a')

# ----- Inline component definition -----
l1 = env.new_line(name='l1', length=10.0,
        components=[
            env.new('qf1', 'Quadrupole', length=1, k1=0.05, at=2.5),
            env.place('qd', at=7.5)])
l1.set_particle_ref('proton', p0c=7000e9)
l1.twiss_default['method'] = '4d'



# ----- Line composed in multiple instructions -----

l2 = env.new_line(name='l2', length=10.0, compose=True)

l2.set_particle_ref('proton', p0c=7000e9)
l2.twiss_default['method'] = '4d'

# Create or install elements
l2.new('qf2', 'Quadrupole', at=2.5, length=1, k1=0.05)
l2.place('qd', at=7.5)

l2.end_compose() # (one day we can do it automatically on twiss, track, etc.)

tw2 = l2.twiss()
