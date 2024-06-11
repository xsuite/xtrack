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
print(line.get_table())

# Shift and tilt selected elements
line['b1'].shift_x = -0.1
line['b1'].rot_s_rad = 0.8
line['b2'].shift_s= 0.2
line['b2'].rot_s_rad = -0.8

tt = line.get_table(attr=True)
tt.cols['s', 'element_type', 'isthick', 'shift_x', 'shift_y', 'shift_s', 'rot_s_rad']
# returns:
#
# name       s element_type isthick shift_x shift_y shift_s rot_s_rad
# seq$start  0 Marker         False       0       0       0         0
# b1         0 Bend            True    -0.1       0       0       0.8
# drift_0    1 Drift           True       0       0       0         0
# b2         2 Bend            True       0       0     0.2      -0.8
# drift_1    3 Drift           True       0       0       0         0
# seq$end    4 Marker         False       0       0       0         0
# _end_point 4                False       0       0       0         0

# Slice the line
slicing_strategies = [
    xt.Strategy(slicing=None),  # Default catch-all
    xt.Strategy(slicing=xt.Teapot(2), element_type=xt.Bend),
]
line.slice_thick_elements(slicing_strategies)
line.build_tracker()

# Inspect
tt = line.get_table(attr=True)
tt.cols['s', 'element_type', 'isthick', 'parent_name',
        'shift_x', 'shift_y', 'shift_s', 'rot_s_rad']
# returns:
#
# Table: 23 rows, 9 cols
# name                 s element_type       isthick parent_name shift_x shift_y shift_s rot_s_rad
# seq$start            0 Marker               False        None       0       0       0         0
# b1_entry             0 Marker               False        None       0       0       0         0
# b1..entry_map        0 ThinSliceBendEntry   False          b1    -0.1       0       0       0.8
# drift_b1..0          0 DriftSliceBend        True          b1       0       0       0         0
# b1..0         0.166667 ThinSliceBend        False          b1    -0.1       0       0       0.8
# drift_b1..1   0.166667 DriftSliceBend        True          b1       0       0       0         0
# b1..1         0.833333 ThinSliceBend        False          b1    -0.1       0       0       0.8
# drift_b1..2   0.833333 DriftSliceBend        True          b1       0       0       0         0
# b1..exit_map         1 ThinSliceBendExit    False          b1    -0.1       0       0       0.8
# b1_exit              1 Marker               False        None       0       0       0         0
# drift_0              1 Drift                 True        None       0       0       0         0
# b2_entry             2 Marker               False        None       0       0       0         0
# b2..entry_map        2 ThinSliceBendEntry   False          b2       0       0     0.2      -0.8
# drift_b2..0          2 DriftSliceBend        True          b2       0       0       0        -0
# b2..0          2.16667 ThinSliceBend        False          b2       0       0     0.2      -0.8
# drift_b2..1    2.16667 DriftSliceBend        True          b2       0       0       0        -0
# b2..1          2.83333 ThinSliceBend        False          b2       0       0     0.2      -0.8
# drift_b2..2    2.83333 DriftSliceBend        True          b2       0       0       0        -0
# b2..exit_map         3 ThinSliceBendExit    False          b2       0       0     0.2      -0.8
# b2_exit              3 Marker               False        None       0       0       0         0
# drift_1              3 Drift                 True        None       0       0       0         0
# seq$end              4 Marker               False        None       0       0       0         0
# _end_point           4                      False        None       0       0       0         0


# Update misalignment for one element. We act on the parent and the effect is
# propagated to the slices.
line['b2'].rot_s_rad = 0.3
line['b2'].shift_x = 2e-3

# Inspect
tt = line.get_table(attr=True)
tt.cols['s', 'element_type', 'isthick', 'parent_name', 'shift_x', 'shift_y', 'rot_s_rad']
# returns:
#
# Table: 23 rows, 8 cols
# name                 s element_type       isthick parent_name shift_x shift_y rot_s_rad
# seq$start            0 Marker               False        None       0       0         0
# b1_entry             0 Marker               False        None       0       0         0
# b1..entry_map        0 ThinSliceBendEntry   False          b1    -0.1       0       0.8
# drift_b1..0          0 DriftSliceBend        True          b1       0       0         0
# b1..0         0.166667 ThinSliceBend        False          b1    -0.1       0       0.8
# drift_b1..1   0.166667 DriftSliceBend        True          b1       0       0         0
# b1..1         0.833333 ThinSliceBend        False          b1    -0.1       0       0.8
# drift_b1..2   0.833333 DriftSliceBend        True          b1       0       0         0
# b1..exit_map         1 ThinSliceBendExit    False          b1    -0.1       0       0.8
# b1_exit              1 Marker               False        None       0       0         0
# drift_0              1 Drift                 True        None       0       0         0
# b2_entry             2 Marker               False        None       0       0         0
# b2..entry_map        2 ThinSliceBendEntry   False          b2   0.002       0       0.3
# drift_b2..0          2 DriftSliceBend        True          b2       0       0         0
# b2..0          2.16667 ThinSliceBend        False          b2   0.002       0       0.3
# drift_b2..1    2.16667 DriftSliceBend        True          b2       0       0         0
# b2..1          2.83333 ThinSliceBend        False          b2   0.002       0       0.3
# drift_b2..2    2.83333 DriftSliceBend        True          b2       0       0         0
# b2..exit_map         3 ThinSliceBendExit    False          b2   0.002       0       0.3
# b2_exit              3 Marker               False        None       0       0         0
# drift_1              3 Drift                 True        None       0       0         0
# seq$end              4 Marker               False        None       0       0         0
# _end_point           4                      False        None       0       0         0