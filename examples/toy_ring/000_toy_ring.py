import numpy as np
import xtrack as xt

# Define elements
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
line = xt.Line(elements=elements,
               element_names=['mqf.1', 'd1.1', 'mb1.1', 'd2.1', # defines the order
                              'mqd.1', 'd3.1', 'mb2.1', 'd4.1',
                              'mqf.2', 'd1.2', 'mb1.2', 'd2.2',
                              'mqd.2', 'd3.2', 'mb2.2', 'd4.2'])
# Define reference particle
line.particle_ref = xt.Particles(p0c=1.2e9, mass0=xt.PROTON_MASS_EV)

#!end-doc-part

line.configure_bend_model(core='full', edge=None)
tw0 = line.twiss(method='4d')

# Print twiss table
tw0.cols['betx bety mux muy'].show(maxrows=None)

# Save to json
line.to_json('toy_ring.json')