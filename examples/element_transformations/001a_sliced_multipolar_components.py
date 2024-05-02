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
print(line.get_table())

# Add multipolar components to elements
line['b1'].knl[2] = 0.001 # Normal sxtupole component
line['q1'].ksl[3] = 0.002 # Skew octupole component

tt = line.get_table(attr=True)
tt.cols['s', 'element_type', 'isthick', 'k2l', 'k3sl'].show()
# prints:
#
# name       s element_type isthick   k2l  k3sl
# seq$start  0 Marker         False     0     0
# b1         0 Bend            True 0.001     0
# drift_0    1 Drift           True     0     0
# q1         2 Quadrupole      True     0 0.002
# drift_1    3 Drift           True     0     0
# seq$end    4 Marker         False     0     0
# _end_point 4                False     0     0

# Slice the line
slicing_strategies = [
    xt.Strategy(slicing=None),  # Default catch-all
    xt.Strategy(slicing=xt.Teapot(2), element_type=xt.Bend),
    xt.Strategy(slicing=xt.Teapot(2), element_type=xt.Quadrupole),
]
line.slice_thick_elements(slicing_strategies)
line.build_tracker()

# Inspect
tt = line.get_table(attr=True)
tt.cols['s', 'element_type', 'isthick', 'parent_name', 'k2l', 'k3sl']

# returns:
#
# Table: 21 rows, 7 cols
# name                 s element_type         isthick parent_name    k2l  k3sl
# seq$start            0 Marker                 False        None      0     0
# b1_entry             0 Marker                 False        None      0     0
# b1..entry_map        0 ThinSliceBendEntry     False          b1      0     0
# drift_b1..0          0 DriftSliceBend          True          b1      0     0
# b1..0         0.166667 ThinSliceBend          False          b1 0.0005     0
# drift_b1..1   0.166667 DriftSliceBend          True          b1      0     0
# b1..1         0.833333 ThinSliceBend          False          b1 0.0005     0
# drift_b1..2   0.833333 DriftSliceBend          True          b1      0     0
# b1..exit_map         1 ThinSliceBendExit      False          b1      0     0
# b1_exit              1 Marker                 False        None      0     0
# drift_0              1 Drift                   True        None      0     0
# q1_entry             2 Marker                 False        None      0     0
# drift_q1..0          2 DriftSliceQuadrupole    True          q1      0     0
# q1..0          2.16667 ThinSliceQuadrupole    False          q1      0 0.001
# drift_q1..1    2.16667 DriftSliceQuadrupole    True          q1      0     0
# q1..1          2.83333 ThinSliceQuadrupole    False          q1      0 0.001
# drift_q1..2    2.83333 DriftSliceQuadrupole    True          q1      0     0
# q1_exit              3 Marker                 False        None      0     0
# drift_1              3 Drift                   True        None      0     0
# seq$end              4 Marker                 False        None      0     0

# Update misalignment for one element. We act on the parent and the effect is
# propagated to the slices.
line['q1'].knl[2] = -0.003
line['q1'].ksl[3] = -0.004

# Inspect
tt = line.get_table(attr=True)
tt.cols['s', 'element_type', 'isthick', 'parent_name', 'k2l', 'k3sl']
# returns:
#
# Table: 21 rows, 7 cols
# name                 s element_type         isthick parent_name     k2l   k3sl
# seq$start            0 Marker                 False        None       0      0
# b1_entry             0 Marker                 False        None       0      0
# b1..entry_map        0 ThinSliceBendEntry     False          b1       0      0
# drift_b1..0          0 DriftSliceBend          True          b1       0      0
# b1..0         0.166667 ThinSliceBend          False          b1  0.0005      0
# drift_b1..1   0.166667 DriftSliceBend          True          b1       0      0
# b1..1         0.833333 ThinSliceBend          False          b1  0.0005      0
# drift_b1..2   0.833333 DriftSliceBend          True          b1       0      0
# b1..exit_map         1 ThinSliceBendExit      False          b1       0      0
# b1_exit              1 Marker                 False        None       0      0
# drift_0              1 Drift                   True        None       0      0
# q1_entry             2 Marker                 False        None       0      0
# drift_q1..0          2 DriftSliceQuadrupole    True          q1       0      0
# q1..0          2.16667 ThinSliceQuadrupole    False          q1 -0.0015 -0.002
# drift_q1..1    2.16667 DriftSliceQuadrupole    True          q1       0      0
# q1..1          2.83333 ThinSliceQuadrupole    False          q1 -0.0015 -0.002
# drift_q1..2    2.83333 DriftSliceQuadrupole    True          q1       0      0
# q1_exit              3 Marker                 False        None       0      0
# drift_1              3 Drift                   True        None       0      0
# seq$end              4 Marker                 False        None       0      0
# _end_point           4                        False        None       0      0