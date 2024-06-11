import xtrack as xt
from cpymad.madx import Madx

# Load a very simple sequence from MAD-X
mad = Madx()
mad.input("""
    seq: sequence, l=4;
    b1: sbend, at=0.5, angle=0.2, l=1;
    b2: sbend, at=2.5, angle=0.3, l=1;
    endsequence;

    beam;
    use,sequence=seq;
""")

line = xt.Line.from_madx_sequence(mad.sequence.seq)
line.build_tracker()

print('The line as imported from MAD-X:')
line.get_table().show()

# Shift and tilt selected elements
line['b1'].shift_x = -0.01
line['b1'].rot_s_rad = 0.8
line['b2'].shift_s = 0.02
line['b2'].rot_s_rad = -0.8

tt = line.get_table(attr=True)
tt.cols['s', 'element_type', 'isthick', 'shift_x', 'shift_y', 'shift_s', 'rot_s_rad']
# returns:
#
# name       s element_type isthick shift_x shift_y shift_s rot_s_rad
# seq$start  0 Marker         False       0       0       0         0
# b1         0 Bend            True   -0.01       0       0       0.8
# drift_0    1 Drift           True       0       0       0         0
# b2         2 Bend            True       0       0    0.02      -0.8
# drift_1    3 Drift           True       0       0       0         0
# seq$end    4 Marker         False       0       0       0         0
# _end_point 4                False       0       0       0         0
