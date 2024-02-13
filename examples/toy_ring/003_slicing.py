import numpy as np
import xtrack as xt

# We build a simple ring
pi = np.pi
lbend = 3
lquad = 0.3
elements = {
    'mqf.1': xt.Quadrupole(length=lquad, k1=0.1),
    'msf.1': xt.Sextupole(length=0.1, k2=0.),
    'd1.1':  xt.Drift(length=0.9),
    'mb1.1': xt.Bend(length=lbend, k0=pi / 2 / lbend, h=pi / 2 / lbend),
    'd2.1':  xt.Drift(length=1),

    'mqd.1': xt.Quadrupole(length=lquad, k1=-0.7),
    'd3.1':  xt.Drift(length=1),
    'mb2.1': xt.Bend(length=lbend, k0=pi / 2 / lbend, h=pi / 2 / lbend),
    'd4.1':  xt.Drift(length=1),

    'mqf.2': xt.Quadrupole(length=lquad, k1=0.1),
    'd1.2':  xt.Drift(length=1),
    'mb1.2': xt.Bend(length=lbend, k0=pi / 2 / lbend, h=pi / 2 / lbend),
    'd2.2':  xt.Drift(length=1),

    'mqd.2': xt.Quadrupole(length=lquad, k1=-0.7),
    'd3.2':  xt.Drift(length=1),
    'mb2.2': xt.Bend(length=lbend, k0=pi / 2 / lbend, h=pi / 2 / lbend),
    'd4.2':  xt.Drift(length=1),
}
line = xt.Line(elements=elements, element_names=list(elements.keys()))
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

ltable = line.get_table(attr=True).cols['s', 'isthick', 'element_type']

# The sextupole msf.1 has one thin slice, as default strategy (1) is applied.
ltable.rows['msf.1_entry':'msf.1_exit']
# returns:
#
# Table: 5 rows, 4 cols
# name              s isthick element_type
# msf.1_entry     0.3   False Marker
# drift_msf.1..0  0.3    True Drift
# msf.1..0       0.35   False Multipole
# drift_msf.1..1 0.35    True Drift
# msf.1_exit      0.4   False Marker

# The bend mb2.1 has three thin slices, as strategy (2) is applied.
ltable.rows['mb2.1_entry':'mb2.1_exit']
# returns:
#
# Table: 7 rows, 4 cols
# name             s isthick element_type
# mb2.1_entry    6.6   False Marker
# drift_mb2.1..0 6.6    True Drift
# mb2.1..0       7.6   False Multipole
# drift_mb2.1..1 7.6    True Drift
# mb2.1..1       8.6   False Multipole
# drift_mb2.1..2 8.6    True Drift
# mb2.1_exit     9.6   False Marker

# The quadrupole mqd.2 has four thin slices, as strategy (3) is applied.
ltable.rows['mqd.2_entry':'mqd.2_exit']
# returns:
#
# Table: 7 rows, 4 cols
# name             s isthick element_type
# mb2.1_entry    6.6   False Marker
# drift_mb2.1..0 6.6    True Drift
# mb2.1..0       7.6   False Multipole
# drift_mb2.1..1 7.6    True Drift
# mb2.1..1       8.6   False Multipole
# drift_mb2.1..2 8.6    True Drift
# mb2.1_exit     9.6   False Marker

# The quadrupole mqf.1 has two thick slices, as strategy (6) is applied.
ltable.rows['mqf.1_entry':'mqf.1_exit']
# returns:
#
# Table: 4 rows, 4 cols
# name              s isthick element_type
# mqf.1_entry       0   False Marker
# mqf.1..0          0    True Quadrupole
# mqf.1..1       0.15    True Quadrupole
# mqf.1_exit      0.3   False Marker

# The quadrupole mqd.1 is left untouched, as strategy (7) is applied.
ltable.rows['mqd.1']
# returns:
#
# Table: 1 row, 4 cols
# name             s isthick element_type
# mqd.1          5.3    True Quadrupole