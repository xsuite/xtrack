from cpymad.madx import Madx
import xtrack as xt
import xobjects as xo
import numpy as np

mad_data = """

cc: crabcavity, l=1, volt=4, freq=400e+06, lag=0.3;

ss: sequence, l=3;
  cc1: cc, at=1;
endsequence;
"""

mad_computation = """
beam, particle=proton, pc=1e3;
use, sequence=ss;
twiss, betx=1, bety=1;
"""
madx = Madx()

madx.input(mad_data)
madx.input(mad_computation)

tw_mad = xt.Table(madx.table.twiss, _copy_cols=True)

line = xt.Line.from_madx_sequence(madx.sequence.ss, deferred_expressions=True)
line.particle_ref = xt.Particles(p0c=1e12)

tw = line.twiss(betx=1, bety=1)
tt = line.get_table(attr=True)
# is:
# Table: 6 rows, 130 cols
# name                   s element_type isthick isreplica ...
# ss$start               0 Marker         False     False
# drift_0                0 Drift           True     False
# cc1                  0.5 CrabCavity      True     False
# drift_1              1.5 Drift           True     False
# ss$end                 3 Marker         False     False
# _end_point             3                False     False


assert np.all(tt.name == np.array(
    ['ss$start', 'drift_0', 'cc1', 'drift_1', 'ss$end', '_end_point']))
xo.assert_allclose(tt.s, np.array([0, 0, 0.5, 1.5, 3, 3]))
assert np.all(tt.element_type == np.array(
    ['Marker', 'Drift', 'CrabCavity', 'Drift', 'Marker', '']))
assert np.all(tt.isthick == np.array([False, True, True, True, False, False]))

xo.assert_allclose(tw.px[-1], tw_mad.px[-1], rtol=0, atol=1e-14)
xo.assert_allclose(tw.py[-1], tw_mad.py[-1], rtol=0, atol=1e-14)
xo.assert_allclose(tw.x[-1], tw_mad.x[-1], rtol=0, atol=1e-14)
xo.assert_allclose(tw.y[-1], tw_mad.y[-1], rtol=0, atol=1e-14)
xo.assert_allclose(tw.ptau[-1], tw_mad.pt[-1], rtol=0, atol=1e-14)

line_slice_thick = line.copy(shallow=True)
line_slice_thick.slice_thick_elements(
    slicing_strategies=[
        xt.Strategy(None),
        xt.Strategy(slicing=xt.Uniform(3, mode='thick'), element_type=xt.CrabCavity),
])
tw_slice_thick = line_slice_thick.twiss(betx=1, bety=1)
tt_slice_thick = line_slice_thick.get_table(attr=True)
# is:
# Table: 10 rows, 130 cols
# name                   s element_type         isthick isreplica ...
# ss$start               0 Marker                 False     False
# drift_0                0 Drift                   True     False
# cc1_entry            0.5 Marker                 False     False
# cc1..0               0.5 ThickSliceCrabCavity    True     False
# cc1..1          0.833333 ThickSliceCrabCavity    True     False
# cc1..2           1.16667 ThickSliceCrabCavity    True     False
# cc1_exit             1.5 Marker                 False     False
# drift_1              1.5 Drift                   True     False
# ss$end                 3 Marker                 False     False
# _end_point             3                        False     False

assert np.all(tt_slice_thick.name == np.array([
    'ss$start', 'drift_0', 'cc1_entry', 'cc1..0', 'cc1..1', 'cc1..2',
    'cc1_exit', 'drift_1', 'ss$end', '_end_point'
]))
xo.assert_allclose(tt_slice_thick.s, np.array([
    0, 0, 0.5, 0.5, 0.83333333, 1.16666667, 1.5, 1.5, 3, 3
]), rtol=0, atol=1e-7)
assert np.all(tt_slice_thick.element_type == np.array([
    'Marker', 'Drift', 'Marker', 'ThickSliceCrabCavity', 'ThickSliceCrabCavity',
    'ThickSliceCrabCavity', 'Marker', 'Drift', 'Marker', ''
]))
assert np.all(tt_slice_thick.isthick == np.array([
    False, True, False, True, True, True, False, True, False, False
]))
# The MAD-X element is implemented with a single kick in the middle, so it is
# normal to see small differences
xo.assert_allclose(tw_slice_thick.px[-1], tw_mad.px[-1], rtol=0, atol=1e-11)
xo.assert_allclose(tw_slice_thick.py[-1], tw_mad.py[-1], rtol=0, atol=1e-11)
xo.assert_allclose(tw_slice_thick.x[-1], tw_mad.x[-1], rtol=0, atol=1e-10)
xo.assert_allclose(tw_slice_thick.y[-1], tw_mad.y[-1], rtol=0, atol=1e-10)
xo.assert_allclose(tw_slice_thick.ptau[-1], tw_mad.pt[-1], rtol=0, atol=1e-5)

line_slice_thin = line.copy(shallow=True)
line_slice_thin.slice_thick_elements(
    slicing_strategies=[
        xt.Strategy(None),
        xt.Strategy(slicing=xt.Uniform(3, mode='thin'), element_type=xt.CrabCavity),
])
tw_slice_thin = line_slice_thin.twiss(betx=1, bety=1)
tt_slice_thin = line_slice_thin.get_table(attr=True)
# is:
# Table: 14 rows, 130 cols
# name                     s element_type         isthick isreplica ...
# ss$start                 0 Marker                 False     False
# drift_0                  0 Drift                   True     False
# cc1_entry              0.5 Marker                 False     False
# drift_cc1..0           0.5 DriftSliceCrabCavity    True     False
# cc1..3                0.75 ThinSliceCrabCavity    False     False
# drift_cc1..1          0.75 DriftSliceCrabCavity    True     False
# cc1..4                   1 ThinSliceCrabCavity    False     False
# drift_cc1..2             1 DriftSliceCrabCavity    True     False
# cc1..5                1.25 ThinSliceCrabCavity    False     False
# drift_cc1..3          1.25 DriftSliceCrabCavity    True     False
# cc1_exit               1.5 Marker                 False     False
# drift_1                1.5 Drift                   True     False
# ss$end                   3 Marker                 False     False
# _end_point               3                        False     False

assert np.all(tt_slice_thin.name == np.array([
    'ss$start', 'drift_0', 'cc1_entry', 'drift_cc1..0', 'cc1..3', 'drift_cc1..1',
    'cc1..4', 'drift_cc1..2', 'cc1..5', 'drift_cc1..3', 'cc1_exit', 'drift_1',
    'ss$end', '_end_point'
]))
xo.assert_allclose(tt_slice_thin.s, np.array([
    0, 0, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.25, 1.25, 1.5, 1.5, 3, 3
]), rtol=0, atol=1e-7)
assert np.all(tt_slice_thin.element_type == np.array([
    'Marker', 'Drift', 'Marker', 'DriftSliceCrabCavity', 'ThinSliceCrabCavity',
    'DriftSliceCrabCavity', 'ThinSliceCrabCavity', 'DriftSliceCrabCavity',
    'ThinSliceCrabCavity', 'DriftSliceCrabCavity', 'Marker', 'Drift', 'Marker', ''
]))
assert np.all(tt_slice_thin.isthick == np.array([
    False, True, False, True, False, True, False, True, False, True, False, True, False, False
]))
xo.assert_allclose(tw_slice_thin.px[-1], tw_mad.px[-1], rtol=0, atol=1e-11)
xo.assert_allclose(tw_slice_thin.py[-1], tw_mad.py[-1], rtol=0, atol=1e-11)
xo.assert_allclose(tw_slice_thin.x[-1], tw_mad.x[-1], rtol=0, atol=1e-10)
xo.assert_allclose(tw_slice_thin.y[-1], tw_mad.y[-1], rtol=0, atol=1e-10)
xo.assert_allclose(tw_slice_thin.ptau[-1], tw_mad.pt[-1], rtol=0, atol=1e-5)