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

#!start-doc-part
hundred_cuts = np.linspace(0, line.get_length(), num=100)
print(f'Make cuts every {hundred_cuts[1]} metres')  # => 21 centimetres
line.cut_at_s(s=list(hundred_cuts))

line.get_table()
# Table: 147 rows, 5 cols
# name               s element_type isthick compound_name
# mqf.1_entry        0 Marker         False mqf.1
# mqf.1..0           0 Quadrupole      True mqf.1
# mqf.1..1    0.214141 Quadrupole      True mqf.1
# mqf.1_exit       0.3 Marker         False mqf.1
# d1.1_entry       0.3 Marker         False d1.1
# d1.1..0          0.3 Drift           True d1.1
# d1.1..1     0.428283 Drift           True d1.1
# d1.1..2     0.642424 Drift           True d1.1
# d1.1..3     0.856566 Drift           True d1.1
# d1.1..4      1.07071 Drift           True d1.1
# d1.1..5      1.28485 Drift           True d1.1
# d1.1_exit        1.3 Marker         False d1.1
# mb1.1_entry      1.3 Marker         False mb1.1
# mb1.1..0         1.3 Bend            True mb1.1
# mb1.1..1     1.49899 Bend            True mb1.1
# mb1.1..2     1.71313 Bend            True mb1.1
# mb1.1..3     1.92727 Bend            True mb1.1
# mb1.1..4     2.14141 Bend            True mb1.1
# mb1.1..5     2.35556 Bend            True mb1.1
# mb1.1..6      2.5697 Bend            True mb1.1
# mb1.1..7     2.78384 Bend            True mb1.1
# mb1.1..8     2.99798 Bend            True mb1.1
# etc...
