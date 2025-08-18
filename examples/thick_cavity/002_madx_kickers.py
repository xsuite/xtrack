from cpymad.madx import Madx
import xtrack as xt

madx = Madx()

mad_data = """

hk: hkicker, l=0.1, kick=1e-3;

ss: sequence, l=3;
  hk1: hk, at=1;
endsequence;
"""

mad_computation = """
beam;
use, sequence=ss;
twiss, betx=1, bety=1;
"""

madx.input(mad_data)
madx.input(mad_computation)

tw_mad = xt.Table(madx.table.twiss, _copy_cols=True)

line = xt.Line.from_madx_sequence(madx.sequence.ss, deferred_expressions=True)
line.particle_ref = xt.Particles(p0c=1e9)

tw = line.twiss(betx=1, bety=1)

env = xt.load(string=mad_data, format='madx')
