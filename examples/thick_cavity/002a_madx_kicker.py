from cpymad.madx import Madx
import xtrack as xt
import xobjects as xo
import numpy as np

loader_mode = 'cpymad' #cpymad/native


mad_data = """

hk: hkicker, l=1, kick=1e-3;

ss: sequence, l=3;
  hk1: hk, at=1;
endsequence;
"""

mad_computation = """
beam;
use, sequence=ss;
twiss, betx=1, bety=1;
"""

madx = Madx()

madx.input(mad_data)
madx.input(mad_computation)

tw_mad = xt.Table(madx.table.twiss, _copy_cols=True)

if loader_mode == 'native':
    env = xt.load(string=mad_data, format='madx')
    line = env['ss']
elif loader_mode == 'cpymad':
    line = xt.Line.from_madx_sequence(madx.sequence.ss, deferred_expressions=True)

line.particle_ref = xt.Particles(p0c=1e9)

tw = line.twiss(betx=1, bety=1)
tt = line.get_table(attr=True)
# is:
# Table: 6 rows, 122 cols
# name                   s element_type isthick isreplica ...
# ss$start               0 Marker         False     False
# drift_0                0 Drift           True     False
# hk1                  0.5 Multipole       True     False
# drift_1              1.5 Drift           True     False
# ss$end                 3 Marker         False     False
# _end_point             3                False     False

assert np.all(tt.name == np.array(
    ['ss$start', 'drift_0', 'hk1', 'drift_1', 'ss$end', '_end_point']))
xo.assert_allclose(tt.s, np.array([0, 0, 0.5, 1.5, 3, 3]))
assert np.all(tt.element_type == np.array(
    ['Marker', 'Drift', 'Multipole', 'Drift', 'Marker', '']))
assert np.all(tt.isthick == np.array([False, True, True, True, False, False]))
assert np.allclose(tt.hkick, np.array([0.   , 0.   , 0.001, 0.   , 0.   , 0.   ]))

xo.assert_allclose(tw.px[-1], tw_mad.px[-1], rtol=0, atol=1e-14)
xo.assert_allclose(tw.py[-1], tw_mad.py[-1], rtol=0, atol=1e-14)
xo.assert_allclose(tw.x[-1], tw_mad.x[-1], rtol=0, atol=1e-14)
xo.assert_allclose(tw.y[-1], tw_mad.y[-1], rtol=0, atol=1e-14)

line_slice_thick = line.copy(shallow=True)
line_slice_thick.slice_thick_elements(
    slicing_strategies=[
        xt.Strategy(None),
        xt.Strategy(slicing=xt.Uniform(3, mode='thick'), element_type=xt.Multipole),
])
tw_slice_thick = line_slice_thick.twiss(betx=1, bety=1)
tt_slice_thick = line_slice_thick.get_table(attr=True)
#is:
# Table: 10 rows, 122 cols
# name                   s element_type        isthick isreplica ...
# ss$start               0 Marker                False     False
# drift_0                0 Drift                  True     False
# hk1_entry            0.5 Marker                False     False
# hk1..0               0.5 ThickSliceMultipole    True     False
# hk1..1          0.833333 ThickSliceMultipole    True     False
# hk1..2           1.16667 ThickSliceMultipole    True     False
# hk1_exit             1.5 Marker                False     False
# drift_1              1.5 Drift                  True     False
# ss$end                 3 Marker                False     False
# _end_point             3                       False     False

assert np.all(tt_slice_thick.name == np.array([
    'ss$start', 'drift_0', 'hk1_entry', 'hk1..0', 'hk1..1', 'hk1..2',
    'hk1_exit', 'drift_1', 'ss$end', '_end_point'
]))
xo.assert_allclose(tt_slice_thick.s, np.array([
    0, 0, 0.5, 0.5, 0.83333333, 1.16666667, 1.5, 1.5, 3, 3
]), rtol=0, atol=1e-7)
assert np.all(tt_slice_thick.element_type == np.array([
    'Marker', 'Drift', 'Marker', 'ThickSliceMultipole', 'ThickSliceMultipole',
    'ThickSliceMultipole', 'Marker', 'Drift', 'Marker', ''
]))
assert np.all(tt_slice_thick.isthick == np.array([
    False, True, False, True, True, True, False, True, False, False
]))
assert np.allclose(tt_slice_thick.hkick, np.array([
    0., 0., 0., 0.001/3, 0.001/3, 0.001/3, 0., 0., 0., 0.
]))
xo.assert_allclose(tw_slice_thick.px[-1], tw_mad.px[-1], rtol=0, atol=1e-14)
xo.assert_allclose(tw_slice_thick.py[-1], tw_mad.py[-1], rtol=0, atol=1e-14)
xo.assert_allclose(tw_slice_thick.x[-1], tw_mad.x[-1], rtol=0, atol=1e-14)
xo.assert_allclose(tw_slice_thick.y[-1], tw_mad.y[-1], rtol=0, atol=1e-14)