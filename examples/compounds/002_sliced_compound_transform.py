import xtrack as xt
from cpymad.madx import Madx

# Carry on with the same example as in 001
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

# Add a transformations
line.transform_compound('b1', x_shift=-0.1, s_rotation=0.8)

# Note that the type of our compound is simply 'Compound' for now, and it has
# the knowledge of the character of the different elements that compose the
# compound:
print(line.get_compound_by_name('b1'))
# Compound(
#     core={'b1_den', 'b1_dex', 'b1'},
#     aperture=set(),
#     entry_transform={'b1_tilt_entry', 'b1_offset_entry'},
#     exit_transform={'b1_tilt_exit', 'b1_offset_exit'},
#     entry={'b1_entry'},
#     exit={'b1_exit'},
# )

# Slice the line
slicing_strategies = [
    xt.Strategy(slicing=None),  # Default catch-all
    xt.Strategy(slicing=xt.Teapot(3), element_type=xt.Bend),
]
line.slice_thick_elements(slicing_strategies)

print(line.get_table())
# Table: 47 rows, 5 cols
# name                   s element_type isthick compound_name
# seq$start              0 Marker         False
# b1_entry               0 Marker         False b1
# b1_offset_entry..0     0 XYShift        False b1 <- Transformations are moved
# b1_tilt_entry..0       0 SRotation      False b1 <- ...to each of the slices
# b1_den                 0 DipoleEdge     False b1
# b1_tilt_exit..0        0 SRotation      False b1 <- ...and undone after
# b1_offset_exit..0      0 XYShift        False b1 <- (here for the edge).
# drift_b1..0            0 Drift           True b1
# b1_offset_entry..1 0.125 XYShift        False b1 <- And here
# b1_tilt_entry..1   0.125 SRotation      False b1 <- ..for the slice
# b1..0              0.125 Multipole      False b1
# b1_tilt_exit..1    0.125 SRotation      False b1 <- ...of the
# b1_offset_exit..1  0.125 XYShift        False b1 <- ...actual bend
# drift_b1..1        0.125 Drift           True b1
#   etc...

# After slicing the compound becomes a 'SlicedCompound' and loses memory of
# its logical structure
print(line.get_compound_by_name('b1'))
# SlicedCompound({'b1_tilt_exit..2', 'drift_b1..0', 'b1..1', ...)

# If we add a transformation of a sliced compound it will now be added around
# it, as is the case for the 'Compound':
line.transform_compound('b2', s_rotation=0.1)

print(line.get_table().rows['b2':'b2':'compound_name'])
# Table: 13 rows, 5 cols
# name                   s element_type isthick compound_name
# b2_tilt_entry          2 SRotation      False b2  <- add the rotation
# b2_entry               2 Marker         False b2
# b2_den                 2 DipoleEdge     False b2
# drift_b2..0            2 Drift           True b2
# b2..0              2.125 Multipole      False b2
# drift_b2..1        2.125 Drift           True b2
# b2..1                2.5 Multipole      False b2
# drift_b2..2          2.5 Drift           True b2
# b2..2              2.875 Multipole      False b2
# drift_b2..3        2.875 Drift           True b2
# b2_dex                 3 DipoleEdge     False b2
# b2_exit                3 Marker         False b2
# b2_tilt_exit           3 SRotation      False b2 <- undo the rotation
