from cpymad.madx import Madx
import xtrack as xt
import xobjects as xo
import numpy as np

mad_data = """

cav: rfcavity, l=1, volt=3, freq=400, lag=0.4;

ss: sequence, l=3;
 cav1: cav, at=1;
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
# Table: 6 rows, 122 cols
# name                   s element_type isthick isreplica ...
# ss$start               0 Marker         False     False
# drift_0                0 Drift           True     False
# cav1                 0.5 Cavity          True     False
# drift_1              1.5 Drift           True     False
# ss$end                 3 Marker         False     False
# _end_point             3                False     False

assert np.all(tt.name == np.array(
    ['ss$start', 'drift_0', 'cav1', 'drift_1', 'ss$end', '_end_point']))
xo.assert_allclose(tt.s, np.array([0, 0, 0.5, 1.5, 3, 3]))
assert np.all(tt.element_type == np.array(
    ['Marker', 'Drift', 'Cavity', 'Drift', 'Marker', '']))
xo.assert_allclose(tt.voltage, np.array(
    [      0.,       0., 3000000.,       0.,       0.,       0.]))
xo.assert_allclose(tt.lag, np.array([0., 0., 144., 0., 0., 0.]))
xo.assert_allclose(tt.frequency, np.array(
    [0.e+00, 0.e+00, 4.e+08, 0.e+00, 0.e+00, 0.e+00]))
assert np.all(tt.isthick == np.array(
    [False, True, True, True, False, False]))

xo.assert_allclose(tw.ptau[-1], tw_mad.pt[-1], rtol=0, atol=1e-9)

line_slice_thick = line.copy(shallow=True)
line_slice_thick.slice_thick_elements(slicing_strategies=[
    xt.Strategy(None),
    xt.Strategy(slicing=xt.Uniform(3, mode='thick'), element_type=xt.Cavity),
])
tw_slice_thick = line_slice_thick.twiss(betx=1, bety=1)
tt_slice_thick = line_slice_thick.get_table(attr=True)
# is:
# Table: 10 rows, 122 cols
# name                   s element_type     isthick isreplica ...
# ss$start               0 Marker             False     False
# drift_0                0 Drift               True     False
# cav1_entry           0.5 Marker             False     False
# cav1..0              0.5 ThickSliceCavity    True     False
# cav1..1         0.833333 ThickSliceCavity    True     False
# cav1..2          1.16667 ThickSliceCavity    True     False
# cav1_exit            1.5 Marker             False     False
# drift_1              1.5 Drift               True     False
# ss$end                 3 Marker             False     False
# _end_point             3                    False     False


assert np.all(tt_slice_thick.name == np.array([
    'ss$start', 'drift_0', 'cav1_entry', 'cav1..0', 'cav1..1', 'cav1..2',
    'cav1_exit', 'drift_1', 'ss$end', '_end_point'
]))
xo.assert_allclose(tt_slice_thick.s, np.array([
    0, 0, 0.5, 0.5, 0.83333333, 1.16666667, 1.5, 1.5, 3, 3
]), rtol=0, atol=1e-7)
assert np.all(tt_slice_thick.element_type == np.array([
    'Marker', 'Drift', 'Marker', 'ThickSliceCavity', 'ThickSliceCavity',
    'ThickSliceCavity', 'Marker', 'Drift', 'Marker', ''
]))
assert np.all(tt_slice_thick.isthick == np.array([
    False, True, False, True, True, True, False, True, False, False
]))
xo.assert_allclose(tt_slice_thick.voltage, np.array([
          0.,       0.,       0., 1000000., 1000000., 1000000.,
          0.,       0.,       0.,       0.]))
xo.assert_allclose(tt_slice_thick.lag, np.array(
    [0., 0., 0., 144., 144., 144., 0., 0., 0., 0.]))
xo.assert_allclose(tt_slice_thick.frequency, np.array([
    0.e+00, 0.e+00, 0.e+00, 4.e+08, 4.e+08, 4.e+08, 0.e+00, 0.e+00, 0.e+00, 0.e+00
]))


xo.assert_allclose(tw_slice_thick.ptau[-1], tw_mad.pt[-1], rtol=0, atol=1e-14)

line_slice_thin = line.copy(shallow=True)
line_slice_thin.slice_thick_elements(slicing_strategies=[
    xt.Strategy(None),
    xt.Strategy(slicing=xt.Uniform(3, mode='thin'), element_type=xt.Cavity),
])
tw_slice_thin = line_slice_thin.twiss(betx=1, bety=1)
tt_slice_thin = line_slice_thin.get_table(attr=True)
# is:
# Table: 14 rows, 124 cols
# name                      s element_type     isthick isreplica ...
# ss$start                  0 Marker             False     False
# drift_0                   0 Drift               True     False
# cav1_entry              0.5 Marker             False     False
# drift_cav1..0           0.5 DriftSliceCavity    True     False
# cav1..3                0.75 ThinSliceCavity    False     False
# drift_cav1..1          0.75 DriftSliceCavity    True     False
# cav1..4                   1 ThinSliceCavity    False     False
# drift_cav1..2             1 DriftSliceCavity    True     False
# cav1..5                1.25 ThinSliceCavity    False     False
# drift_cav1..3          1.25 DriftSliceCavity    True     False
# cav1_exit               1.5 Marker             False     False
# drift_1                 1.5 Drift               True     False
# ss$end                    3 Marker             False     False
# _end_point                3                    False     False

assert np.all(tt_slice_thin.name == np.array([
    'ss$start', 'drift_0', 'cav1_entry', 'drift_cav1..0', 'cav1..3', 'drift_cav1..1',
    'cav1..4', 'drift_cav1..2', 'cav1..5', 'drift_cav1..3', 'cav1_exit', 'drift_1',
    'ss$end', '_end_point'
]))
xo.assert_allclose(tt_slice_thin.s, np.array([
    0, 0, 0.5, 0.5, 0.75, 0.75, 1, 1, 1.25, 1.25, 1.5, 1.5, 3, 3
]), rtol=0, atol=1e-7)
assert np.all(tt_slice_thin.element_type == np.array([
    'Marker', 'Drift', 'Marker', 'DriftSliceCavity', 'ThinSliceCavity',
    'DriftSliceCavity', 'ThinSliceCavity', 'DriftSliceCavity', 'ThinSliceCavity',
    'DriftSliceCavity', 'Marker', 'Drift', 'Marker', ''
]))
assert np.all(tt_slice_thin.isthick == np.array([
    False, True, False, True, False, True, False, True, False, True, False, True, False, False
]))
xo.assert_allclose(tt_slice_thin.voltage, np.array([
             0.,       0.,       0.,       0., 1000000.,       0.,
       1000000.,       0., 1000000.,       0.,       0.,       0.,
             0.,       0.]))
xo.assert_allclose(tt_slice_thin.frequency, np.array([
    0.e+00, 0.e+00, 0.e+00, 0.e+00, 4.e+08, 0.e+00, 4.e+08, 0.e+00,
    4.e+08, 0.e+00, 0.e+00, 0.e+00, 0.e+00, 0.e+00
]))
xo.assert_allclose(tt_slice_thin.lag, np.array([
    0., 0., 0., 0., 144., 0., 144., 0., 144., 0., 0., 0., 0., 0.
]))
xo.assert_allclose(tw_slice_thick.ptau[-1], tw_mad.pt[-1], rtol=0, atol=1e-14)

