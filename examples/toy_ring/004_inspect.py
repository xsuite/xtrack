import numpy as np
import xtrack as xt

pi = np.pi
lbend = 3
elements = {
    'mqf.1': xt.Quadrupole(length=0.3, k1=0.1),
    'd1.1':  xt.Drift(length=1),
    'mb1.1': xt.Bend(length=lbend, k0=pi / 2 / lbend, h=pi / 2 / lbend),
    'd2.1':  xt.Drift(length=1),

    'mqd.1': xt.Quadrupole(length=0.3, k1=-0.7),
    'd3.1':  xt.Drift(length=1),
    'mb2.1': xt.Bend(length=lbend, k0=pi / 2 / lbend, h=pi / 2 / lbend),
    'd4.1':  xt.Drift(length=1),

    'mqf.2': xt.Quadrupole(length=0.3, k1=0.1),
    'd1.2':  xt.Drift(length=1),
    'mb1.2': xt.Bend(length=lbend, k0=pi / 2 / lbend, h=pi / 2 / lbend),
    'd2.2':  xt.Drift(length=1),

    'mqd.2': xt.Quadrupole(length=0.3, k1=-0.7),
    'd3.2':  xt.Drift(length=1),
    'mb2.2': xt.Bend(length=lbend, k0=pi / 2 / lbend, h=pi / 2 / lbend),
    'd4.2':  xt.Drift(length=1),
}

# Build the ring
line = xt.Line(elements=elements, element_names=list(elements.keys()))
line.particle_ref = xt.Particles(p0c=1.2e9, mass0=xt.PROTON_MASS_EV)
line.build_tracker()

# Quick access to an element and its attributes (by name)
line['mqf.1'] # is Quadrupole(length=0.3, k1=0.1, ...)
line['mqf.1'].k1 # is 0.1
line['mqf.1'].length # is 0.3

# Quick access to an element and its attributes (by index)
line[0] # is Quadrupole(length=0.3, k1=0.1, ...)
line[0].k1 # is 0.1
line[0].length # is 0.3

# Tuple with all element Names
line.element_names # is ('mqf.1', 'd1.1', 'mb1.1', 'd2.1', 'mqd.1', ...

# Tuple with all element objects
line.elements # is (Quadrupole(length=0.3, k1=0.1, ...), Drift(length=1), ...