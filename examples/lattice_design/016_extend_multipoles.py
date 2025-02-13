import xtrack as xt

env = xt.Environment()

env.vars.default_to_zero = True
line = env.new_line(components=[
    env.new('b1', xt.Bend, length=1, knl=['a', 'b', 'c'], ksl=['d', 'e', 'f']),
    env.new('q1', xt.Quadrupole, length=1, knl=['a', 'b', 'c'], ksl=['d', 'e', 'f']),
    env.new('s1', xt.Sextupole, length=1, knl=['a', 'b', 'c'], ksl=['d', 'e', 'f']),
    env.new('o1', xt.Octupole, length=1, knl=['a', 'b', 'c'], ksl=['d', 'e', 'f']),
    env.new('s2', xt.Solenoid, length=1, knl=['a', 'b', 'c'], ksl=['d', 'e', 'f']),
    env.new('m1', xt.Multipole, length=1, knl=['a', 'b', 'c'], ksl=['d', 'e', 'f']),
])

# Set the strengths
env['a'] = 1
env['b'] = 2
env['c'] = 3
env['d'] = 4
env['e'] = 5
env['f'] = 6

# Extend knl and ksl for selected elements
line.extend_knl_ksl(order=10, element_names=['b1', 'q1'])

env['b1'].knl # is [1., 2., 3., 0., 0., 0., 0., 0., 0., 0., 0.]

# Extend knl and ksl for all elements
line.extend_knl_ksl(order=10)

env['s2'].knl # is [1., 2., 3., 0., 0., 0., 0., 0., 0., 0., 0.]
