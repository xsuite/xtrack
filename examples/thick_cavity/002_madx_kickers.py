from cpymad.madx import Madx
import xtrack as xt

madx = Madx()

# madx.call("../../test_data/hllhc15_thick/lhc.seq")
# madx.call("../../test_data/hllhc15_thick/hllhc_sequence.madx")
# madx.call("../../test_data/hllhc15_thick/opt_round_150_1500.madx")

# madx.beam()
# madx.use('lhcb1')
# madx.twiss()
# tw = xt.Table(madx.table.twiss, _copy_cols=True)


    
mad_data = """

hk: hkicker, l=0.1, kick=1e-3;

ss: sequence, l=3;
  hk1: hk, at=1;
endsequence;

beam;
use, sequence=ss;
twiss, betx=1, bety=1;
"""

madx.input(mad_data)

tw_mad = xt.Table(madx.table.twiss, _copy_cols=True)

line = xt.Line.from_madx_sequence(madx.sequence.ss, deferred_expressions=True)
line.particle_ref = xt.Particles(p0c=1e9)

tw = line.twiss(betx=1, bety=1)

env = xt.load(string=mad_data, format='madx')
