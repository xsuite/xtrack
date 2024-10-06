import numpy as np
import xtrack as xt
pi = np.pi

# We create an environment
env = xt.Environment()

# We define variables that we will use for controlling the length and the
# integrated strength of the focusing quadrupoles
env['l.quad'] = 0.3
env['k1l.qf.1'] = 0
env['k1l.qf.2'] = 0

# Expressions can be associated to any beam element property, when creating the
# element. For example:
lbend = 3
line = env.new_line(components=[
    env.new('mqf.1', xt.Quadrupole, length=0.3, k1='k1l.qf.1 / l.quad'),
    env.new('d1.1',  xt.Drift, length=1),
    env.new('mb1.1', xt.Bend, length=lbend, k0=pi / 2 / lbend, h=pi / 2 / lbend),
    env.new('d2.1',  xt.Drift, length=1),

    env.new('mqd.1', xt.Quadrupole, length=0.3, k1=0), # k1 will be set later
    env.new('d3.1',  xt.Drift, length=1),
    env.new('mb2.1', xt.Bend, length=lbend, k0=pi / 2 / lbend, h=pi / 2 / lbend),
    env.new('d4.1',  xt.Drift, length=1),

    env.new('mqf.2', xt.Quadrupole, length=0.3, k1='k1l.qf.2 / l.quad'),
    env.new('d1.2',  xt.Drift, length=1),
    env.new('mb1.2', xt.Bend, length=lbend, k0=pi / 2 / lbend, h=pi / 2 / lbend),
    env.new('d2.2',  xt.Drift, length=1),

    env.new('mqd.2', xt.Quadrupole, length=0.3, k1=0), # k1 will be set later
    env.new('d3.2',  xt.Drift, length=1),
    env.new('mb2.2', xt.Bend, length=lbend, k0=pi / 2 / lbend, h=pi / 2 / lbend),
    env.new('d4.2',  xt.Drift, length=1),
])
line.particle_ref = xt.Particles(p0c=1.2e9, mass0=xt.PROTON_MASS_EV)

# Expressions can also be assigned after the creation of the line. For example, we
# can set the integrated strength of the defocusing quadrupoles (note that line[...]
# is equivalent to env[...]):
line['k1l.qd.1'] = -0.1
line['k1l.qd.2'] = -0.1
line['mqd.1'].k1 = 'k1l.qd.1 / l.quad'
line['mqd.2'].k1 = 'k1l.qd.2 / l.quad'

# When a variable is changed, the corresponding element property is automatically
# updated:
line.vars['k1l.qf.1'] = 0.1
line['mqf.1'].k1 # is 0.333, i.e. 0.1 / lquad

# Expressions can be modified after the creation of the line. For example, we can
# create a variable controlling the integrated strength of the two focusing quadrupoles
line['k1lf'] = 0.1
line['k1l.qf.1'] = 'k1lf'
line['k1l.qf.2'] = 'k1lf'
# and a variable controlling the integrated strength of the two defocusing quadrupoles
line['k1ld'] = -0.7
line['k1l.qd.1'] = 'k1ld'
line['k1l.qd.2'] = 'k1ld'

# Changes on the controlling variable are propagated to the two controlled ones and
# to the corresponding element properties:
line['k1lf'] = 0.2
line['k1l.qf.1'] # is 0.2
line['k1l.qf.2'] # is 0.2
line['mqf.1'].k1 # is 0.666, i.e. 0.2 / lquad
line['mqf.2'].k1 # is 0.666, i.e. 0.2 / lquad

# The `_info()` method of a variable provides information on the existing relations
# between the variables. For example:
line.info('k1l.qf.1')
# prints:
##  vars['k1l.qf.1']._get_value()
#   vars['k1l.qf.1'] = 0.2
#
##  vars['k1l.qf.1']._expr
#   vars['k1l.qf.1'] = vars['k1lf']
#
##  vars['k1l.qf.1']._expr._get_dependencies()
#   vars['k1lf'] = 0.2
#
##  vars['k1l.qf.1']._find_dependant_targets()
#   element_refs['mqf.1'].k1

line.info('k1lf')
# prints:
##  vars['k1lf']._get_value()
#   vars['k1lf'] = 0.2
#
##  vars['k1lf']._expr is None
#
##  vars['k1lf']._find_dependant_targets()
#   vars['k1l.qf.2']
#   element_refs['mqf.2'].k1
#   vars['k1l.qf.1']
#   element_refs['mqf.1'].k1

line['mqf.1'].get_info('k1')
# prints:
##  element_refs['mqf.1'].k1._get_value()
#   element_refs['mqf.1'].k1 = 0.6666666666666667
#
##  element_refs['mqf.1'].k1._expr
#   element_refs['mqf.1'].k1 = (vars['k1l.qf.1'] / 0.3)
#
##  element_refs['mqf.1'].k1._expr._get_dependencies()
#   vars['k1l.qf.1'] = 0.2
#
##  element_refs['mqf.1'].k1 does not influence any target

# Expressions can include multiple variables and mathematical operations. For example
line['a'] = '3 * sqrt(k1lf) + 2 * k1ld'

# The `line.vars.get_table()` method returns a table with the value of all the
# existing variables:
line.vars.get_table()
# returns:
#
# Table: 9 rows, 2 cols
# name     value
# t_turn_s     0
# k1l.qf.1   0.2
# k1l.qd.1  -0.7
# k1l.qf.2   0.2
# k1l.qd.2  -0.7
# k1lf       0.2
# k1ld      -0.7
# a            4
# b            6

# Regular expressions can be used to select variables. For example we can select all
# the variables containing `qf` using the following:
var_tab = line.vars.get_table()
var_tab.rows['.*qf.*']
# returns:
#
# Table: 2 rows, 2 cols
# name     value
# k1l.qf.1   0.2
# k1l.qf.2   0.2
