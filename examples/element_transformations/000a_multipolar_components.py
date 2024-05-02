import xtrack as xt
from cpymad.madx import Madx

# Load a very simple sequence from MAD-X
mad = Madx()
mad.input("""
    seq: sequence, l=4;
    b1: sbend, at=0.5, angle=0.2, l=1;
    q1: quadrupole, at=2.5, k1=0.1, l=1;
    endsequence;

    beam;
    use,sequence=seq;
""")

line = xt.Line.from_madx_sequence(mad.sequence.seq)
line.build_tracker()

print('The line as imported from MAD-X:')
line.get_table().show()

# Add multipolar components to elements
line['b1'].knl[2] = 0.001 # Normal sxtupole component
line['q1'].ksl[3] = 0.002 # Skew octupole component

tt = line.get_table(attr=True)
tt.cols['s', 'element_type', 'isthick', 'k2l', 'k3sl'].show()
# returns:
#
# name       s element_type isthick   k2l  k3sl
# seq$start  0 Marker         False     0     0
# b1         0 Bend            True 0.001     0
# drift_0    1 Drift           True     0     0
# q1         2 Quadrupole      True     0 0.002
# drift_1    3 Drift           True     0     0
# seq$end    4 Marker         False     0     0
# _end_point 4                False     0     0
