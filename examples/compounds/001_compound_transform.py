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

# The MAD-X elements b1 and b2 will be exploded into smaller more specialised
# elements in Xsuite: i.e. edges, the core of the bend, markers, etc.
print('The line as imported from MAD-X:')
print(line.get_table())

print('Specified transformations are added to the compound:')
line.transform_compound('b1', x_shift=-0.1, s_rotation=0.8)
line.transform_compound('b2', y_shift=0.2, s_rotation=-0.8)

print(line.get_table())
# Table: 23 rows, 5 cols
# name            s element_type isthick compound_name
# seq$start       0 Marker         False
# b1_offset_entry 0 XYShift        False b1  <- added shift
# b1_tilt_entry   0 SRotation      False b1  <- added tilt
# b1_entry        0 Marker         False b1
# b1_den          0 DipoleEdge     False b1
# b1              0 Bend            True b1
# b1_dex          1 DipoleEdge     False b1
# b1_exit         1 Marker         False b1
# b1_tilt_exit    1 SRotation      False b1  <- undo tilt
# b1_offset_exit  1 XYShift        False b1  <- undo shift
# drift_0         1 Drift           True
# b2_offset_entry 2 XYShift        False b2  <- added shift
# b2_tilt_entry   2 SRotation      False b2  <- added tilt
# b2_entry        2 Marker         False b2
# b2_den          2 DipoleEdge     False b2
# b2              2 Bend            True b2
# b2_dex          3 DipoleEdge     False b2
# b2_exit         3 Marker         False b2
# b2_tilt_exit    3 SRotation      False b2  <- undo tilt
# b2_offset_exit  3 XYShift        False b2  <- undo shift
# drift_1         3 Drift           True
# seq$end         4 Marker         False
# _end_point      4                False

print('Further transformations are added on top of the current ones:')
line.transform_compound('b1', y_shift=0.1)

print(line.get_table().rows['b1':'b1':'compound_name'])
# Table: 11 rows, 5 cols
# name              s element_type isthick compound_name
# b1_offset_entry_1 0 XYShift        False b1  <- added new shift
# b1_offset_entry   0 XYShift        False b1
# b1_tilt_entry     0 SRotation      False b1
# b1_entry          0 Marker         False b1
# b1_den            0 DipoleEdge     False b1
# b1                0 Bend            True b1
# b1_dex            1 DipoleEdge     False b1
# b1_exit           1 Marker         False b1
# b1_tilt_exit      1 SRotation      False b1
# b1_offset_exit    1 XYShift        False b1
# b1_offset_exit_1  1 XYShift        False b1  <- undo new shift
