import xtrack as xt

env = xt.Environment()

# We can associate a reference particle to the environment
# which will be passed to all lines generated from this environment
env.partice_ref = xt.Particles(p0c=2e9, mass0=xt.PROTON_MASS_EV)

#############
# Variables #
#############

# The environment can be used to create and inspect variables and associated
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

# We can use the eval method to evaluate the value for a given expression:
env.eval('sqrt(b + a)') # returns 3.0

# We can inspect the value and expression of all variables in the environment
env.vars.get_table()
# returns:
#
# Table: 3 rows, 3 cols
# name             value expr
# t_turn_s             0 None
# a                    3 None
# b                    6 (2.0 * vars['a'])

############
# Elements #
############

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

# It is also possible to add to the environment elements that are instantiated by
# the user:
mysext = xt.Sextupole(length=1.0, k2=0.1)
env.elements['ms'] = mysext

env['ms'] # accesses the sextupole
env.info('ms')

# prints:
# Element of type:  Sextupole
# k2                  0.1                                     None
# k2s                 0.0                                     None
# length              1.0                                     None
# order               5                                       None
# inv_factorial_order 0.008333333333333333                    None
# knl                 [0. 0. 0. 0. 0. 0.]                     None
# ...

#########
# Lines #
#########

# The environment can be used to create and access lines

line1 = env.new_line(
    name='l1', components=['mq', env.new('dd', xt.Drift, length=10)])

line2 = env.new_line(
    name='l2', components=[
        env.place('dd'), # place already created elements
        env.new('ip', xt.Marker)])

env['l1'] # accesses the first line
env['l2'] # accesses the second line

# The environment is associated to the line
line1.env # is env
line2.env # is env

# The variables and elements of the environment are shared by all the lines
# created by the environment
line2['a'] # is 3.0
line2['a'] = 4.0
env['a'] # is 4.0

# All methods available on the environment are also available on the lines, e.g.:
line2.info('a') # is equivalent to env.info('a')
