import numpy as np
import xtrack as xt

# We build a simple ring
pi = np.pi
lbend = 3
lquad = 0.3
elements = {
    'mqf.1': xt.Quadrupole(length=lquad, k1=0.1),
    'd1.1':  xt.Drift(length=1),
    'mb1.1': xt.Bend(length=lbend, k0=pi / 2 / lbend, h=pi / 2 / lbend),
    'd2.1':  xt.Drift(length=1),

    'mqd.1': xt.Quadrupole(length=lquad, k1=-0.7),
    'd3.1':  xt.Drift(length=1),
    'mb2.1': xt.Bend(length=lbend, k0=pi / 2 / lbend, h=pi / 2 / lbend),
    'd4.1':  xt.Drift(length=1),

    'mqf.2': xt.Quadrupole(length=lquad, k1=0.1),
    'd1.2':  xt.Drift(length=1),
    'mb1.2': xt.Bend(length=lbend, k0=pi / 2 / lbend, h=pi / 2 / lbend),
    'd2.2':  xt.Drift(length=1),

    'mqd.2': xt.Quadrupole(length=lquad, k1=-0.7),
    'd3.2':  xt.Drift(length=1),
    'mb2.2': xt.Bend(length=lbend, k0=pi / 2 / lbend, h=pi / 2 / lbend),
    'd4.2':  xt.Drift(length=1),
}
line = xt.Line(elements=elements, element_names=list(elements.keys()))
line.particle_ref = xt.Particles(p0c=1.2e9, mass0=xt.PROTON_MASS_EV)

# For each quadrupole we create a variable controlling its integrated strength.
# Expressions can be associated to any beam element property, using the `element_refs`
# attribute of the line. For example:
line.vars['k1l.qf.1'] = 0
line.element_refs['mqf.1'].k1 = line.vars['k1l.qf.1'] / lquad
line.vars['k1l.qd.1'] = 0
line.element_refs['mqd.1'].k1 = line.vars['k1l.qd.1'] / lquad
line.vars['k1l.qf.2'] = 0
line.element_refs['mqf.2'].k1 = line.vars['k1l.qf.2'] / lquad
line.vars['k1l.qd.2'] = 0
line.element_refs['mqd.2'].k1 = line.vars['k1l.qd.2'] / lquad

# When a variable is changed, the corresponding element property is automatically
# updated:
line.vars['k1l.qf.1'] = 0.1
line['mqf.1'].k1 # is 0.333, i.e. 0.1 / lquad

# We can create a variable controlling the integrated strength of the two
# focusing quadrupoles
line.vars['k1lf'] = 0.1
line.vars['k1l.qf.1'] = line.vars['k1lf']
line.vars['k1l.qf.2'] = line.vars['k1lf']
# and a variable controlling the integrated strength of the two defocusing quadrupoles
line.vars['k1ld'] = -0.7
line.vars['k1l.qd.1'] = line.vars['k1ld']
line.vars['k1l.qd.2'] = line.vars['k1ld']

# Changes on the controlling variable are propagated to the two controlled ones and
# to the corresponding element properties:
line.vars['k1lf'] = 0.2
line.vars['k1l.qf.1']._get_value() # is 0.2
line.vars['k1l.qf.2']._get_value() # is 0.2
line['mqf.1'].k1 # is 0.666, i.e. 0.2 / lquad
line['mqf.2'].k1 # is 0.666, i.e. 0.2 / lquad

# The `_info()` method of a variable provides information on the existing relations
# between the variables. For example:
line.vars['k1l.qf.1']._info()
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

line.vars['k1lf']._info()
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

line.element_refs['mqf.1'].k1._info()
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
line.vars['a'] = 3 * line.functions.sqrt(line.vars['k1lf']) + 2 * line.vars['k1ld']

# As seen above, line.vars['varname'] returns a reference object that
# can be used to build further references, or to inspect its properties.
# To get the current value of the variable, one needs to use `._get_value()`
# For quick access to the current value of a variable, one can use the `line.varval`
# attribute or its shortcut `line.vv`:
line.varval['k1lf'] # is 0.2
line.vv['k1lf']     # is 0.2
# Note an important difference when using `line.vars` or `line.varval` in building
# expressions. For example:
line.vars['a'] = 3.
line.vars['b'] = 2 * line.vars['a']
# In this case the reference to the quantity `line.vars['a']` is stored in the
# expression, and the value of `line.vars['b']` is updated when `line.vars['a']`
# changes:
line.vars['a'] = 4.
line.vv['b'] # is 8.
# On the contrary, when using `line.varval` or `line.vv` in building expressions,
# the current value of the variable is stored in the expression:
line.vv['a'] = 3.
line.vv['b'] = 2 * line.vv['a']
line.vv['b'] # is 6.
line.vv['a'] = 4.
line.vv['b'] # is still 6.

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
