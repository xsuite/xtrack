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
