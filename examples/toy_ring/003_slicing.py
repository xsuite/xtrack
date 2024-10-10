import numpy as np
import xtrack as xt

# We build a simple ring
pi = np.pi
lbend = 3
lquad = 0.3
env = xt.Environment()
line = env.new_line(components=[
    env.new('mqf.1', xt.Quadrupole, length=lquad, k1=0.1),
    env.new('msf.1', xt.Sextupole, length=0.1, k2=0.02),
    env.new('d1.1',  xt.Drift, length=0.9),
    env.new('mb1.1', xt.Bend, length=lbend, k0=pi / 2 / lbend, h=pi / 2 / lbend),
    env.new('d2.1',  xt.Drift, length=1),

    env.new('mqd.1', xt.Quadrupole, length=lquad, k1=-0.7),
    env.new('d3.1',  xt.Drift, length=1),
    env.new('mb2.1', xt.Bend, length=lbend, k0=pi / 2 / lbend, h=pi / 2 / lbend),
    env.new('d4.1',  xt.Drift, length=1),

    env.new('mqf.2', xt.Quadrupole, length=lquad, k1=0.1),
    env.new('d1.2',  xt.Drift, length=1),
    env.new('mb1.2', xt.Bend, length=lbend, k0=pi / 2 / lbend, h=pi / 2 / lbend),
    env.new('d2.2',  xt.Drift, length=1),

    env.new('mqd.2', xt.Quadrupole, length=lquad, k1=-0.7),
    env.new('d3.2',  xt.Drift, length=1),
    env.new('mb2.2', xt.Bend, length=lbend, k0=pi / 2 / lbend, h=pi / 2 / lbend),
    env.new('d4.2',  xt.Drift, length=1),
])
line.particle_ref = xt.Particles(p0c=1.2e9, mass0=xt.PROTON_MASS_EV)

line_before_slicing = line.copy() # Keep for comparison

# Slice different elements with different strategies (in case multiple strategies
# apply to the same element, the last one takes precedence)
line.slice_thick_elements(
    slicing_strategies=[
        # Slicing with thin elements
        xt.Strategy(slicing=xt.Teapot(1)), # (1) Default applied to all elements
        xt.Strategy(slicing=xt.Uniform(2), element_type=xt.Bend), # (2) Selection by element type
        xt.Strategy(slicing=xt.Teapot(3), element_type=xt.Quadrupole),  # (4) Selection by element type
        xt.Strategy(slicing=xt.Teapot(4), name='mb1.*'), # (5) Selection by name pattern
        # Slicing with thick elements
        xt.Strategy(slicing=xt.Uniform(2, mode='thick'), name='mqf.*'), # (6) Selection by name pattern
        # Do not slice (leave untouched)
        xt.Strategy(slicing=None, name='mqd.1') # (7) Selection by name
    ])
line.build_tracker()
line_before_slicing.build_tracker()

# Ispect the result:

ltable = line.get_table(attr=True).cols['s', 'isthick', 'element_type',
                                        'parent_name', 'k0l', 'k1l', 'k2l']

# The sextupole msf.1 has one thin slice, as default strategy (1) is applied.
ltable.rows['msf.1_entry':'msf.1_exit']
# returns:
#
# Table: 5 rows, 8 cols
# name                s isthick element_type         parent_name k0l k1l   k2l
# msf.1_entry       0.3   False Marker                      None   0   0     0
# drift_msf.1..0    0.3    True DriftSliceSextupole        msf.1   0   0     0
# msf.1..0         0.35   False ThinSliceSextupole         msf.1   0   0 0.002
# drift_msf.1..1   0.35    True DriftSliceSextupole        msf.1   0   0     0
# msf.1_exit        0.4   False Marker                      None   0   0     0

# The bend mb2.1 has three thin slices, as strategy (2) is applied.
ltable.rows['mb2.1_entry':'mb2.1_exit']
# returns:
#
# Table: 9 rows, 8 cols
# name               s isthick element_type         parent_name      k0l k1l k2l
# mb2.1_entry      6.6   False Marker                      None        0   0   0
# mb2.1..entry_map 6.6   False ThinSliceBendEntry         mb2.1        0   0   0
# drift_mb2.1..0   6.6    True DriftSliceBend             mb2.1        0   0   0
# mb2.1..0         7.6   False ThinSliceBend              mb2.1 0.785398   0   0
# drift_mb2.1..1   7.6    True DriftSliceBend             mb2.1        0   0   0
# mb2.1..1         8.6   False ThinSliceBend              mb2.1 0.785398   0   0
# drift_mb2.1..2   8.6    True DriftSliceBend             mb2.1        0   0   0
# mb2.1..exit_map  9.6   False ThinSliceBendExit          mb2.1        0   0   0
# mb2.1_exit       9.6   False Marker                      None        0   0   0

# The quadrupole mqd.2 has four thin slices, as strategy (3) is applied.
ltable.rows['mqd.2_entry':'mqd.2_exit']
# returns:
#
# Table: 9 rows, 8 cols
# name                   s isthick element_type         parent_name k0l   k1l ...
# mqd.2_entry         15.9   False Marker                      None   0     0
# drift_mqd.2..0      15.9    True DriftSliceQuadrupole       mqd.2   0     0
# mqd.2..0         15.9375   False ThinSliceQuadrupole        mqd.2   0 -0.07
# drift_mqd.2..1   15.9375    True DriftSliceQuadrupole       mqd.2   0     0
# mqd.2..1           16.05   False ThinSliceQuadrupole        mqd.2   0 -0.07
# drift_mqd.2..2     16.05    True DriftSliceQuadrupole       mqd.2   0     0
# mqd.2..2         16.1625   False ThinSliceQuadrupole        mqd.2   0 -0.07
# drift_mqd.2..3   16.1625    True DriftSliceQuadrupole       mqd.2   0     0
# mqd.2_exit          16.2   False Marker                      None   0     0

# The quadrupole mqf.1 has two thick slices, as strategy (6) is applied.
ltable.rows['mqf.1_entry':'mqf.1_exit']
# returns:
#
# Table: 4 rows, 8 cols
# name                s isthick element_type         parent_name k0l   k1l k2l
# mqf.1_entry         0   False Marker                      None   0     0   0
# mqf.1..0            0    True ThickSliceQuadrupole       mqf.1   0 0.015   0
# mqf.1..1         0.15    True ThickSliceQuadrupole       mqf.1   0 0.015   0
# mqf.1_exit        0.3   False Marker                      None   0     0   0

# The quadrupole mqd.1 is left untouched, as strategy (7) is applied.
ltable.rows['mqd.1']
# returns:
#
# Table: 1 row, 8 cols
# name               s isthick element_type         parent_name k0l   k1l k2l
# mqd.1            5.3    True Quadrupole                  None   0 -0.21   0


########################################
# Change properties of sliced elements #
########################################

# Sliced elements are updated whenever their parent is changed. For example:

# Inspect a quadrupole:
ltable.rows['mqf.1.*']
# returns:
#
# Table: 4 rows, 8 cols
# name                s isthick element_type         parent_name k0l   k1l k2l
# mqf.1_entry         0   False Marker                      None   0     0   0
# mqf.1..0            0    True ThickSliceQuadrupole       mqf.1   0 0.015   0
# mqf.1..1         0.15    True ThickSliceQuadrupole       mqf.1   0 0.015   0
# mqf.1_exit        0.3   False Marker                      None   0     0   0

# Change the the strength of the parent
line['mqf.1'].k1 = 0.2

# Inspect
ltable = line.get_table(attr=True).cols['s', 'isthick', 'element_type',
                                        'parent_name', 'k0l', 'k1l', 'k2l']
ltable.rows['mqf.1.*']
# returns (the strength of the slices has changed):
#
# Table: 4 rows, 8 cols
# name                s isthick element_type         parent_name k0l  k1l k2l
# mqf.1_entry         0   False Marker                      None   0    0   0
# mqf.1..0            0    True ThickSliceQuadrupole       mqf.1   0 0.03   0
# mqf.1..1         0.15    True ThickSliceQuadrupole       mqf.1   0 0.03   0
# mqf.1_exit        0.3   False Marker                      None   0    0   0
