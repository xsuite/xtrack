import xtrack as xt

env = xt.Environment()

# We can associate a reference particle to the environment
# which will be passed to all lines generated from this environment
env.partice_ref = xt.Particles(p0c=2e9, mass0=xt.PROTON_MASS_EV)

# The environment can be used to create and inspect veriables and associated
# deferred expressions
env['a'] = 3.
env['b'] = '2 * a'

env['b'] # is 6.0
env.get_expr('b') # is (2.0 * vars['a'])
env.info('b')
# prints:
#  vars['b']._get_value()
#  vars['b'] = 6.0

#  vars['b']._expr
#  vars['b'] = (2.0 * vars['a'])

#  vars['b']._expr._get_dependencies()
#  vars['a'] = 3.0

#  vars['b'] does not influence any target

# The environment can be used to create and inspect elements
env.new('mq', xt.Quadrupole, length='5*a', k1='b')

env['mq'] # accesses the quadrupole

env.info('mq')
# prints:
# Element of type:  Quadrupole
# k1                  6.0                              vars['b']
# k1s                 0.0                              None
# length              15.0                             (5.0 * vars['a'])
# num_multipole_kicks 0                                None
# order               5                                None
# inv_factorial_order 0.008333333333333333             None
# knl                 [0. 0. 0. 0. 0. 0.]              None
# ...

env['mq'].length # is 15.0
env['mq'].k1 # is 6.0

env['mq'].get_expr('length') # is (5.0 * vars['a'])
env['mq'].get_info('length')
# prints:
#  element_refs['mq'].length._get_value()
#  element_refs['mq'].length = 15.0
#
#  element_refs['mq'].length._expr
#  element_refs['mq'].length = (5.0 * vars['a'])
#
#  element_refs['mq'].length._expr._get_dependencies()
#  vars['a'] = 3.0
#
#  element_refs['mq'].length does not influence any target