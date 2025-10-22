import xtrack as xt

env = xt.Environment()
env['a'] = 1.

env.new('qd', 'Quadrupole', length=1, k1='-0.05 * a')

# ----- Line with inline component definition -----

l1 = env.new_line(name='l1', length=10.0,
        components=[
            env.new('qf1', 'Quadrupole', length=1, k1=0.05, at=2.5),
            env.place('qd', at=7.5)])
l1.set_particle_ref('proton', p0c=7000e9)
l1.twiss_default['method'] = '4d'


# ----- Line composed in multiple instructions -----

l2 = env.new_line(name='l2', length=10.0, compose=True)

# Optionally set properties while in compose mode
l2.set_particle_ref('proton', p0c=7000e9)
l2.twiss_default['method'] = '4d'

# Create or install elements
l2.new('qf2', 'Quadrupole', at=2.5, length=1, k1=0.05)
l2.place('qd', at=7.5)

tw2 = l2.twiss() # ends the compose mode

l2.place('qd', at=8.5)    # OK
l2.place('qf1', at=1.5)  # OK

tw3 = l2.twiss()

# ----- more details -----

l2.mode # is 'compose'
# The topology of the line is given by line.composer
# line.element_names cannot be edited (if it is there, it's a tuple)

l2.end_compose()

l2.mode # is 'normal'
# The topology of the line is given by line.element_names
# As usual, after line.discard_tracker(), line.element_names is a list
# and can be edited

