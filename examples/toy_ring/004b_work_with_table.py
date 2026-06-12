import numpy as np
import xtrack as xt

pi = np.pi
lbend = 3

# Build a simple ring
env = xt.Environment()
line = env.new_line(components=[
    env.new('mqf.1', xt.Quadrupole, length=0.3, k1=0.1),
    env.new('d1.1',  xt.Drift, length=1),
    env.new('mb1.1', xt.Bend, length=lbend, angle=pi / 2),
    env.new('d2.1',  xt.Drift, length=1),

    env.new('mqd.1', xt.Quadrupole, length=0.3, k1=-0.7),
    env.new('d3.1',  xt.Drift, length=1),
    env.new('mb2.1', xt.Bend, length=lbend, angle=pi / 2),
    env.new('d4.1',  xt.Drift, length=1),

    env.new('mqf.2', xt.Quadrupole, length=0.3, k1=0.1),
    env.new('d1.2',  xt.Drift, length=1),
    env.new('mb1.2', xt.Bend, length=lbend, angle=pi / 2),
    env.new('d2.2',  xt.Drift, length=1),

    env.new('mqd.2', xt.Quadrupole, length=0.3, k1=-0.7),
    env.new('d3.2',  xt.Drift, length=1),
    env.new('mb2.2', xt.Bend, length=lbend, angle=pi / 2),
    env.new('d4.2',  xt.Drift, length=1),
])
line.set_particle_ref('proton', p0c=1.2e9)

# Get the twiss table of the line with the strengths
tab = line.twiss4d(strengths=True)

# The table can be printed
tab.show()
# prints:
#
# name                   s             x            px             y            py ...
# mqf.1                  0   5.27539e-10   7.91903e-12             0             0
# d1.1                 0.3   5.27539e-10  -7.91903e-12             0             0
# mb1.1                1.3    5.1962e-10  -7.91903e-12             0             0
# d2.1                 4.3   4.27314e-10  -4.04126e-11             0             0
# mqd.1                5.3   3.86901e-10  -4.04126e-11             0             0
# d3.1                 5.6   3.86901e-10   4.04127e-11             0             0
# mb2.1                6.6   4.27314e-10   4.04127e-11             0             0
# d4.1                 9.6   5.19621e-10   7.91896e-12             0             0
# mqf.2               10.6    5.2754e-10   7.91896e-12             0             0
# d1.2                10.9    5.2754e-10   -7.9191e-12             0             0
# mb1.2               11.9    5.1962e-10   -7.9191e-12             0             0
# d2.2                14.9   4.27314e-10  -4.04126e-11             0             0
# mqd.2               15.9   3.86901e-10  -4.04126e-11             0             0
# d3.2                16.2   3.86901e-10   4.04127e-11             0             0
# mb2.2               17.2   4.27314e-10   4.04127e-11             0             0
# d4.2                20.2   5.19621e-10   7.91901e-12             0             0
# _end_point          21.2    5.2754e-10   7.91901e-12             0             0

# Access to a single element of the table
tab['s', 'mb2.1'] # is 6.6

# Access to a single column of the table
tab['s'] # is [0.0, 0.3, 1.3, 4.3, 5.3, 5.6, 6.6, 9.6, 10.6, 10.9, 11.9, ...

# Access to selected columns (the output is a Table object)
tab.cols['betx bety alfx alfy']
# returns:
#
# TwissTable: 17 rows, 5 cols
# name                betx          bety          alfx          alfy
# mqf.1            1.27738       4.79104     0.0997349      0.103198
# d1.1             1.27738       4.79104    -0.0997349     -0.103198
# mb1.1            2.26749       5.20838     -0.890376     -0.314144
# d2.1             2.88391       8.99176      0.890376     -0.946981
# mqd.1            1.72481       11.0967      0.268732      -1.15793
# d3.1             1.72481       11.0967     -0.268732       1.15793
# mb2.1            2.88391       8.99176     -0.890376      0.946981
# d4.1             2.26749       5.20838      0.890376      0.314144
# mqf.2            1.27738       4.79104     0.0997349      0.103198
# d1.2             1.27738       4.79104    -0.0997349     -0.103198
# mb1.2            2.26749       5.20838     -0.890376     -0.314144
# d2.2             2.88391       8.99176      0.890376     -0.946981
# mqd.2            1.72481       11.0967      0.268732      -1.15793
# d3.2             1.72481       11.0967     -0.268732       1.15793
# mb2.2            2.88391       8.99176     -0.890376      0.946981
# d4.2             2.26749       5.20838      0.890376      0.314144
# _end_point       1.27738       4.79104     0.0997349      0.103198

# Simple expressions can be used in the cols
tab.cols['betx dx dx/sqrt(betx)']
# returns:
#
# TwissTable: 17 rows, 4 cols
# name                betx            dx dx/sqrt(betx)
# mqf.1            1.27738       2.27721       2.01486
# d1.1             1.27738       2.27721       2.01486
# mb1.1            2.26749       2.24303       1.48958
# d2.1             2.88391       1.84457       1.08619
# mqd.1            1.72481       1.67012       1.27168
# d3.1             1.72481       1.67012       1.27168
# mb2.1            2.88391       1.84457       1.08619
# d4.1             2.26749       2.24303       1.48958
# mqf.2            1.27738       2.27721       2.01486
# d1.2             1.27738       2.27721       2.01486
# mb1.2            2.26749       2.24303       1.48958
# d2.2             2.88391       1.84457       1.08619
# mqd.2            1.72481       1.67012       1.27168
# d3.2             1.72481       1.67012       1.27168
# mb2.2            2.88391       1.84457       1.08619
# d4.2             2.26749       2.24303       1.48958
# _end_point       1.27738       2.27721       2.01486

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

 # The output of this operation is a table, hence row selections

# Elements can be selected by type using the match search (applicable to any column)
tab.rows.match(element_type='Quadrupole')
# returns:
#
# Table: 4 rows, 94 cols
# name          s element_type isthick isreplica parent_name iscollective
# mqf.1         0 Quadrupole      True     False        None        False
# mqd.1       5.3 Quadrupole      True     False        None        False
# mqf.2      10.6 Quadrupole      True     False        None        False
# mqd.2      15.9 Quadrupole      True     False        None        False

# Match supports regular expressions
tab.rows.match(element_type='Quad.*|Be.*')
# returns:
# LineTable: 8 rows, 186 cols
# name              s element_type isthick isreplica parent_name ...
# mqf.1             0 Quadrupole      True     False None       
# mb1.1           1.3 Bend            True     False None       
# mqd.1           5.3 Quadrupole      True     False None       
# mb2.1           6.6 Bend            True     False None       
# mqf.2          10.6 Quadrupole      True     False None       
# mb1.2          11.9 Bend            True     False None       
# mqd.2          15.9 Quadrupole      True     False None       
# mb2.2          17.2 Bend            True     False None   

# A section of the table can be selected using names
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
tab.rows['mqd.1<<3':'mb2.1>>2']
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
