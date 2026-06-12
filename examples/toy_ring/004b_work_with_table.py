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
# name       element_type             s          betx          bety ...
# mqf.1      Quadrupole               0       1.27738       4.79104
# d1.1       Drift                  0.3       1.27738       4.79104
# mb1.1      Bend                   1.3       2.26749       5.20838
# d2.1       Drift                  4.3       2.88391       8.99176
# mqd.1      Quadrupole             5.3       1.72481       11.0967
# d3.1       Drift                  5.6       1.72481       11.0967
# mb2.1      Bend                   6.6       2.88391       8.99176
# d4.1       Drift                  9.6       2.26749       5.20838
# mqf.2      Quadrupole            10.6       1.27738       4.79104
# d1.2       Drift                 10.9       1.27738       4.79104
# mb1.2      Bend                  11.9       2.26749       5.20838
# d2.2       Drift                 14.9       2.88391       8.99176
# mqd.2      Quadrupole            15.9       1.72481       11.0967
# d3.2       Drift                 16.2       1.72481       11.0967
# mb2.2      Bend                  17.2       2.88391       8.99176
# d4.2       Drift                 20.2       2.26749       5.20838
# _end_point                       21.2       1.27738       4.79104

# Single values can be accessed using the column name and the row name. For example:
tab['s', 'mb2.1'] # is 6.6

# Entire columns can be accessed using the column name. For example:
tab['s'] # is [0.0, 0.3, 1.3, 4.3, 5.3, 5.6, 6.6, 9.6, 10.6, 10.9, 11.9, ...

# The `.cols` attribute can be used to access multiple columns (the output is 
# a Table object). For example:
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

# The .rows attribute can be used access selected rows
tab.rows[['mqf.1', 'mqd.1']]
# returns:
#
# TwissTable: 2 rows, 98 cols
# TwissTable: 2 rows, 98 cols
# name  element_type             s          betx          bety ...
# mqf.1 Quadrupole               0       1.27738       4.79104
# mqd.1 Quadrupole             5.3       1.72481       11.0967

# Regular expressions can be used to select rows
tab.rows['mb.*']
# returns:
#
# TwissTable: 4 rows, 98 cols
# name  element_type             s          betx          bety ...
# mb1.1 Bend                   1.3       2.26749       5.20838
# mb2.1 Bend                   6.6       2.88391       8.99176
# mb1.2 Bend                  11.9       2.26749       5.20838
# mb2.2 Bend                  17.2       2.88391       8.99176

# Elements can be selected by type using the match search (applicable to any column)
tab.rows.match(element_type='Quadrupole')
# returns:
#
# TwissTable: 4 rows, 98 cols
# name  element_type             s          betx          bety ...
# mqf.1 Quadrupole               0       1.27738       4.79104
# mqd.1 Quadrupole             5.3       1.72481       11.0967
# mqf.2 Quadrupole            10.6       1.27738       4.79104
# mqd.2 Quadrupole            15.9       1.72481       11.0967

# rows.match supports regular expressions
tab.rows.match(element_type='Quad.*|Be.*')
# returns:
# TwissTable: 8 rows, 98 cols
# name  element_type             s          betx          bety ...
# mqf.1 Quadrupole               0       1.27738       4.79104
# mb1.1 Bend                   1.3       2.26749       5.20838
# mqd.1 Quadrupole             5.3       1.72481       11.0967
# mb2.1 Bend                   6.6       2.88391       8.99176
# mqf.2 Quadrupole            10.6       1.27738       4.79104
# mb1.2 Bend                  11.9       2.26749       5.20838
# mqd.2 Quadrupole            15.9       1.72481       11.0967
# mb2.2 Bend                  17.2       2.88391       8.99176

# rows.match_not can be used to select rows not matching a given condition.
# For example to select all elements that are not drifts:
tab.rows.match_not(element_type='Drift')
#
# TwissTable: 9 rows, 98 cols
# name  element_type             s          betx          bety ...
# mqd.1 Quadrupole             5.3       1.72481       11.0967
# d3.1  Drift                  5.6       1.72481       11.0967
# mb2.1 Bend                   6.6       2.88391       8.99176
# d4.1  Drift                  9.6       2.26749       5.20838
# mqf.2 Quadrupole            10.6       1.27738       4.79104
# d1.2  Drift                 10.9       1.27738       4.79104
# mb1.2 Bend                  11.9       2.26749       5.20838
# d2.2  Drift                 14.9       2.88391       8.99176
# mqd.2 Quadrupole            15.9       1.72481       11.0967

# A section of the table can be selected using names
tab.rows['mqd.1':'mqd.2']
# returns:
#
# TwissTable: 9 rows, 98 cols
# name  element_type             s          betx          bety ...
# mqd.1 Quadrupole             5.3       1.72481       11.0967
# d3.1  Drift                  5.6       1.72481       11.0967
# mb2.1 Bend                   6.6       2.88391       8.99176
# d4.1  Drift                  9.6       2.26749       5.20838
# mqf.2 Quadrupole            10.6       1.27738       4.79104
# d1.2  Drift                 10.9       1.27738       4.79104
# mb1.2 Bend                  11.9       2.26749       5.20838
# d2.2  Drift                 14.9       2.88391       8.99176
# mqd.2 Quadrupole            15.9       1.72481       11.0967

# A section of the ring can be selected using the s coordinate
tab.rows[3.0:7.0:'s']
# returns:
#
# TwissTable: 4 rows, 98 cols
# name  element_type             s          betx          bety ...
# d2.1  Drift                  4.3       2.88391       8.99176
# mqd.1 Quadrupole             5.3       1.72481       11.0967
# d3.1  Drift                  5.6       1.72481       11.0967
# mb2.1 Bend                   6.6       2.88391       8.99176

# A section of the table can be selected using indexes relative one element
# (e.g. to get from three elements upstream of 'mqd.1' to two elements
# downstream of 'mb2.1')
tab.rows['mqd.1<<3':'mb2.1>>2']
# returns:
#
# TwissTable: 8 rows, 98 cols
# name  element_type             s          betx          bety ...
# d1.1  Drift                  0.3       1.27738       4.79104
# mb1.1 Bend                   1.3       2.26749       5.20838
# d2.1  Drift                  4.3       2.88391       8.99176
# mqd.1 Quadrupole             5.3       1.72481       11.0967
# d3.1  Drift                  5.6       1.72481       11.0967
# mb2.1 Bend                   6.6       2.88391       8.99176
# d4.1  Drift                  9.6       2.26749       5.20838
# mqf.2 Quadrupole            10.6       1.27738       4.79104

# As rows and column selectors return Table objects they can be chained for example
# in the following we select rows in the range 'd1.1'-'d2.2', which are not of type Drift,
# and that match the regular expression 'mb.*' and we select the columns `betx` and `bety`:
tab.rows['d1.1':'d2.2'].rows.match_not(element_type='Drift').rows.match(name='mb.*').cols['betx bety']
# returns:
#
# TwissTable: 3 rows, 3 cols
# name           betx          bety
# mb1.1       2.26749       5.20838
# mb2.1       2.88391       8.99176
# mb1.2       2.26749       5.20838
