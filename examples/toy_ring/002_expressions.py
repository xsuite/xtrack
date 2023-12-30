import numpy as np
import xtrack as xt

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

# Build the ring
line = xt.Line(elements=elements, element_names=list(elements.keys()))
line.particle_ref = xt.Particles(p0c=1.2e9, mass0=xt.PROTON_MASS_EV)

# For each quadrupole create a variable controlling its integrated strength
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
line.vars['k1l.qf'] = 0.1
line.vars['k1l.qf.1'] = line.vars['k1l.qf']
line.vars['k1l.qf.2'] = line.vars['k1l.qf']
# and a variable controlling the integrated strength of the two defocusing quadrupoles
line.vars['k1l.qd'] = -0.7
line.vars['k1l.qd.1'] = line.vars['k1l.qd']
line.vars['k1l.qd.2'] = line.vars['k1l.qd']

# Changes on this variable are propagated to the two controlled variables and
# to the corresponding element properties:
line.vars['k1l.qf'] = 0.2
line.vars['k1l.qf.1']._value # is 0.
line.vars['k1l.qf.2']._value # is 0.
line['mqf.1'].k1 # is 0.666, i.e. 0.2 / lquad
line['mqf.2'].k1 # is 0.666, i.e. 0.2 / lquad




