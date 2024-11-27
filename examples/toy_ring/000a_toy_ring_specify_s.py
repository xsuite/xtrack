import numpy as np
import xtrack as xt

pi = np.pi
lbend = 3
l_halfcell = 5.3

# Create an environment
env = xt.Environment()

# Build a simple ring
line = env.new_line(components=[
    env.new('mqf.1', xt.Quadrupole, length=0.3, k1=0.1, at=0.15),
    env.new('d1.1',  xt.Drift, length=1),
    env.new('mb1.1', xt.Bend, length=lbend, k0=pi / 2 / lbend, h=pi / 2 / lbend),
        # drift between mb1.1 and mdq.1 defined implicitly
    env.new('mqd.1', xt.Quadrupole, length=0.3, k1=-0.7, at=l_halfcell, from_='mqf.1'),
    env.new('d3.1',  xt.Drift, length=1),
    env.new('mb2.1', xt.Bend, length=lbend, k0=pi / 2 / lbend, h=pi / 2 / lbend),
        # drift between mb2.1 and mdq.1 defined implicitly
    env.new('mqf.2', xt.Quadrupole, length=0.3, k1=0.1, at=l_halfcell, from_='mqd.1'),
    env.new('d1.2',  xt.Drift, length=1),
    env.new('mb1.2', xt.Bend, length=lbend, k0=pi / 2 / lbend, h=pi / 2 / lbend),
        # drift between mb1.2 and mdq.2 defined implicitly
    env.new('mqd.2', xt.Quadrupole, length=0.3, k1=-0.7, at=l_halfcell, from_='mqf.2'),
    env.new('d3.2',  xt.Drift, length=1),
    env.new('mb2.2', xt.Bend, length=lbend, k0=pi / 2 / lbend, h=pi / 2 / lbend),
    env.new('d4.2',  xt.Drift, length=1),
])

# Define reference particle
line.particle_ref = xt.Particles(p0c=1.2e9, mass0=xt.PROTON_MASS_EV)

# Print the line table
line.get_table().show()
# prints:
#
# name                   s element_type isthick isreplica parent_name ...
# mqf.1                  0 Quadrupole      True     False None
# d1.1                 0.3 Drift           True     False None
# mb1.1                1.3 Bend            True     False None
# d2.1                 4.3 Drift           True     False None
# etc.

#!end-doc-part

line.configure_bend_model(core='full', edge=None)
tw0 = line.twiss(method='4d')

# Print twiss table
print('\nTwiss:')
tw0.cols['betx bety mux muy'].show()

# Save to json
line.to_json('toy_ring.json')