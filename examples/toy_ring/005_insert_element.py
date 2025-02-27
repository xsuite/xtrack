import numpy as np
import xtrack as xt

pi = np.pi
lbend = 3

# Create an environment
env = xt.Environment()

# Build a simple ring
line = env.new_line(components=[
    env.new('mqf.1', xt.Quadrupole, length=0.3, k1=0.1),
    env.new('d1.1',  xt.Drift, length=1),
    env.new('mb1.1', xt.Bend, length=lbend, k0=pi / 2 / lbend, h=pi / 2 / lbend),
    env.new('d2.1',  xt.Drift, length=1),

    env.new('mqd.1', xt.Quadrupole, length=0.3, k1=-0.7),
    env.new('d3.1',  xt.Drift, length=1),
    env.new('mb2.1', xt.Bend, length=lbend, k0=pi / 2 / lbend, h=pi / 2 / lbend),
    env.new('d4.1',  xt.Drift, length=1),

    env.new('mqf.2', xt.Quadrupole, length=0.3, k1=0.1),
    env.new('d1.2',  xt.Drift, length=1),
    env.new('mb1.2', xt.Bend, length=lbend, k0=pi / 2 / lbend, h=pi / 2 / lbend),
    env.new('d2.2',  xt.Drift, length=1),

    env.new('mqd.2', xt.Quadrupole, length=0.3, k1=-0.7),
    env.new('d3.2',  xt.Drift, length=1),
    env.new('mb2.2', xt.Bend, length=lbend, k0=pi / 2 / lbend, h=pi / 2 / lbend),
    env.new('d4.2',  xt.Drift, length=1),
])

# Build the ring
line.particle_ref = xt.Particles(p0c=1.2e9, mass0=xt.PROTON_MASS_EV)
line.build_tracker()

# Inspect the line
tab = line.get_table()
tab.show()
# prints:
#
# name          s element_type isthick isreplica parent_name iscollective
# mqf.1         0 Quadrupole      True     False        None        False
# d1.1        0.3 Drift           True     False        None        False
# mb1.1       1.3 Bend            True     False        None        False
# d2.1        4.3 Drift           True     False        None        False
# mqd.1       5.3 Quadrupole      True     False        None        False
# d3.1        5.6 Drift           True     False        None        False
# mb2.1       6.6 Bend            True     False        None        False
# d4.1        9.6 Drift           True     False        None        False
# mqf.2      10.6 Quadrupole      True     False        None        False
# d1.2       10.9 Drift           True     False        None        False
# mb1.2      11.9 Bend            True     False        None        False
# d2.2       14.9 Drift           True     False        None        False
# mqd.2      15.9 Quadrupole      True     False        None        False
# d3.2       16.2 Drift           True     False        None        False
# mb2.2      17.2 Bend            True     False        None        False
# d4.2       20.2 Drift           True     False        None        False
# _end_point 21.2                False     False        None        False

# Define a sextupole
my_sext = xt.Sextupole(length=0.1, k2=0.1)
# Insert copies of the defined sextupole downstream of the quadrupoles
line.discard_tracker() # needed to modify the line structure
line.insert('msf.1', my_sext.copy(), anchor='start', at='mqf.1@end')
line.insert('msd.1', my_sext.copy(), anchor='start', at='mqd.1@end')
line.insert('msf.2', my_sext.copy(), anchor='start', at='mqf.2@end')
line.insert('msd.2', my_sext.copy(), anchor='start', at='mqd.2@end')

# Define a rectangular aperture
my_aper = xt.LimitRect(min_x=-0.02, max_x=0.02, min_y=-0.01, max_y=0.01)
# Insert the aperture upstream of the first bending magnet
line.insert('aper', my_aper, at='mb1.1@start')

line.get_table().show()
# prints:
#
# name          s element_type isthick isreplica parent_name iscollective
# mqf.1         0 Quadrupole      True     False        None        False
# d1.1..0     0.3 DriftSlice      True     False        d1.1        False
# msf.1       0.4 Sextupole       True     False        None        False
# d1.1..2     0.5 DriftSlice      True     False        d1.1        False
# aper        1.3 LimitRect      False     False        None        False
# mb1.1       1.3 Bend            True     False        None        False
# d2.1        4.3 Drift           True     False        None        False
# mqd.1       5.3 Quadrupole      True     False        None        False
# d3.1..0     5.6 DriftSlice      True     False        d3.1        False
# msd.1       5.7 Sextupole       True     False        None        False
# d3.1..2     5.8 DriftSlice      True     False        d3.1        False
# mb2.1       6.6 Bend            True     False        None        False
# d4.1        9.6 Drift           True     False        None        False
# mqf.2      10.6 Quadrupole      True     False        None        False
# d1.2..0    10.9 DriftSlice      True     False        d1.2        False
# msf.2        11 Sextupole       True     False        None        False
# d1.2..2    11.1 DriftSlice      True     False        d1.2        False
# mb1.2      11.9 Bend            True     False        None        False
# d2.2       14.9 Drift           True     False        None        False
# mqd.2      15.9 Quadrupole      True     False        None        False
# d3.2..0    16.2 DriftSlice      True     False        d3.2        False
# msd.2      16.3 Sextupole       True     False        None        False
# d3.2..2    16.4 DriftSlice      True     False        d3.2        False
# mb2.2      17.2 Bend            True     False        None        False
# d4.2       20.2 Drift           True     False        None        False
# _end_point 21.2                False     False        None        False
