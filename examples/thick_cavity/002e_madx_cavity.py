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
beam, particle=proton, pc=1;
use, sequence=ss;
twiss, betx=1, bety=1;
"""

madx = Madx()

madx.input(mad_data)
madx.input(mad_computation)

tw_mad = xt.Table(madx.table.twiss, _copy_cols=True)

line = xt.Line.from_madx_sequence(madx.sequence.ss, deferred_expressions=True)

line.particle_ref = xt.Particles(p0c=1e9)

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
assert np.all(tt.isthick == np.array(
    [False, True, True, True, False, False]))

xo.assert_allclose(tw.ptau[-1], tw_mad.pt[-1], rtol=0, atol=1e-9)

line_slice_thick = line.copy(shallow=True)
line_slice_thick.slice_thick_elements(
    xt.Strategy(None),
    xt.Strategy(slicing=xt.Uniform(3, mode='thick'), element_type=xt.Cavity),
)
tw_slice_thick = line_slice_thick.twiss(betx=1, bety=1)
tt_slice_thick = line_slice_thick.get_table(attr=True)
# is: