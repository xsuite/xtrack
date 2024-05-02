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

# Tuple with all element names
line.element_names # is ('mqf.1', 'd1.1', 'mb1.1', 'd2.1', 'mqd.1', ...

# Tuple with all element objects
line.elements # is (Quadrupole(length=0.3, k1=0.1, ...), Drift(length=1), ...

# `line.attr[...]` can be used for efficient extraction of a given attribute for
# all elements. For example:
line.attr['length'] # is (0.3, 1, 3, 1, 0.3, 1, 3, 1, 0.3, 1, 3, 1, 0.3, 1, 3, 1)
line.attr['k1l'] # is ('0.03, 0.0, 0.0, 0.0, -0.21, 0.0, 0.0, 0.0, 0.03, ... )

# The list of all attributes can be found in
line.attr.keys() # is ('length', 'k1', 'k1l', 'k2', 'k2l', 'k3', 'k3l', 'k4', ...

# `line.get_table()`` can be used to get a table with information about the line
# elements. For example:
tab = line.get_table()

# The table can be printed
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

# Access to a single element of the table
tab['s', 'mb2.1'] # is 6.6

# Access to a single column of the table
tab['s'] # is [0.0, 0.3, 1.3, 4.3, 5.3, 5.6, 6.6, 9.6, 10.6, 10.9, 11.9, ...

# Regular expressions can be used to select elements by name
tab.rows['mb.*']
# returns:
#
# Table: 4 rows, 94 cols
# name          s element_type isthick isreplica parent_name iscollective
# mb1.1       1.3 Bend            True     False        None        False
# mb2.1       6.6 Bend            True     False        None        False
# mb1.2      11.9 Bend            True     False        None        False
# mb2.2      17.2 Bend            True     False        None        False


# Elements can be selected by type
tab.rows[tab.element_type == 'Quadrupole']
# returns:
#
# Table: 4 rows, 94 cols
# name          s element_type isthick isreplica parent_name iscollective
# mqf.1         0 Quadrupole      True     False        None        False
# mqd.1       5.3 Quadrupole      True     False        None        False
# mqf.2      10.6 Quadrupole      True     False        None        False
# mqd.2      15.9 Quadrupole      True     False        None        False

# A section of the ring can be selected using names
tab.rows['mqd.1':'mqd.2']
# returns:
#
# Table: 9 rows, 94 cols
# name          s element_type isthick isreplica parent_name iscollective
# mqd.1       5.3 Quadrupole      True     False        None        False
# d3.1        5.6 Drift           True     False        None        False
# mb2.1       6.6 Bend            True     False        None        False
# d4.1        9.6 Drift           True     False        None        False
# mqf.2      10.6 Quadrupole      True     False        None        False
# d1.2       10.9 Drift           True     False        None        False
# mb1.2      11.9 Bend            True     False        None        False
# d2.2       14.9 Drift           True     False        None        False
# mqd.2      15.9 Quadrupole      True     False        None        False

# A section of the ring can be selected using the s coordinate
tab.rows[3.0:7.0:'s']
# returns:
#
# Table: 4 rows, 94 cols
# name         s element_type isthick isreplica parent_name iscollective
# d2.1       4.3 Drift           True     False        None        False
# mqd.1      5.3 Quadrupole      True     False        None        False
# d3.1       5.6 Drift           True     False        None        False
# mb2.1      6.6 Bend            True     False        None        False

# A section of the ring can be selected using indexes relative one element
# (e.g. to get from three elements upstream of 'mqd.1' to two elements
# downstream of 'mb2.1')
tab.rows['mqd.1%%-3':'mb2.1%%2']
# returns:
#
# Table: 8 rows, 94 cols
# name          s element_type isthick isreplica parent_name iscollective
# d1.1        0.3 Drift           True     False        None        False
# mb1.1       1.3 Bend            True     False        None        False
# d2.1        4.3 Drift           True     False        None        False
# mqd.1       5.3 Quadrupole      True     False        None        False
# d3.1        5.6 Drift           True     False        None        False
# mb2.1       6.6 Bend            True     False        None        False
# d4.1        9.6 Drift           True     False        None        False
# mqf.2      10.6 Quadrupole      True     False        None        False

# Each of the selection methods above returns a valid table, hence selections
# can be chained. For example:
tab.rows[0:10:'s'].rows['mb.*']
# returns:
#
# Table: 2 rows, 94 cols
# name         s element_type isthick isreplica parent_name iscollective
# mb1.1      1.3 Bend            True     False        None        False
# mb2.1      6.6 Bend            True     False        None        False

# All attributes extracted by `line.attr[...]` can be included in the table
# using `attr=True`. For example, using `tab.cols[...]` to select columns, we
# we can get the focusing strength of all quadrupoles in the ring:
tab = line.get_table(attr=True)
tab.rows[tab.element_type=='Quadrupole'].cols['s length k1l']
# returns:
#
# Table: 4 rows, 4 cols
# name          s length   k1l
# mqf.1         0    0.3  0.03
# mqd.1       5.3    0.3 -0.21
# mqf.2      10.6    0.3  0.03
# mqd.2      15.9    0.3 -0.21
