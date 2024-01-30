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

# Inspect the line
tab = line.get_table()
tab.show()
# prints:
#
# Table: 17 rows, 5 cols
# name          s element_type isthick compound_name
# mqf.1         0 Quadrupole      True
# d1.1        0.3 Drift           True
# mb1.1       1.3 Bend            True
# d2.1        4.3 Drift           True
# mqd.1       5.3 Quadrupole      True
# d3.1        5.6 Drift           True
# mb2.1       6.6 Bend            True
# d4.1        9.6 Drift           True
# mqf.2      10.6 Quadrupole      True
# d1.2       10.9 Drift           True
# mb1.2      11.9 Bend            True
# d2.2       14.9 Drift           True
# mqd.2      15.9 Quadrupole      True
# d3.2       16.2 Drift           True
# mb2.2      17.2 Bend            True
# d4.2       20.2 Drift           True
# _end_point 21.2                False

# Define a sextupole
my_sext = xt.Sextupole(length=0.1, k2=0.1)
# Insert copies of the defined sextupole downstream of the quadrupoles
line.discard_tracker() # needed to modify the line structure
line.insert_element('msf.1', my_sext.copy(), at_s=tab['s', 'mqf.1'] + 0.4)
line.insert_element('msd.1', my_sext.copy(), at_s=tab['s', 'mqd.1'] + 0.4)
line.insert_element('msf.2', my_sext.copy(), at_s=tab['s', 'mqf.2'] + 0.4)
line.insert_element('msd.2', my_sext.copy(), at_s=tab['s', 'mqd.2'] + 0.4)

# Define a rectangular aperture
my_aper = xt.LimitRect(min_x=-0.02, max_x=0.02, min_y=-0.01, max_y=0.01)
# Insert the aperture upstream of the first bending magnet
line.insert_element('aper', my_aper, index='mb1.1')

line.get_table().show()
# prints:
#
# name          s element_type isthick compound_name
# mqf.1         0 Quadrupole      True
# d1.1_u      0.3 Drift           True
# msf.1       0.4 Sextupole       True
# d1.1_d      0.5 Drift           True
# aper        1.3 LimitRect      False
# mb1.1       1.3 Bend            True
# d2.1        4.3 Drift           True
# mqd.1       5.3 Quadrupole      True
# d3.1_u      5.6 Drift           True
# msd.1       5.7 Sextupole       True
# d3.1_d      5.8 Drift           True
# mb2.1       6.6 Bend            True
# d4.1        9.6 Drift           True
# mqf.2      10.6 Quadrupole      True
# d1.2_u     10.9 Drift           True
# msf.2        11 Sextupole       True
# d1.2_d     11.1 Drift           True
# mb1.2      11.9 Bend            True
# d2.2       14.9 Drift           True
# mqd.2      15.9 Quadrupole      True
# d3.2_u     16.2 Drift           True
# msd.2      16.3 Sextupole       True
# d3.2_d     16.4 Drift           True
# mb2.2      17.2 Bend            True
# d4.2       20.2 Drift           True
# _end_point 21.2                False
