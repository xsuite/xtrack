import xtrack as xt
from cpymad.madx import Madx

mad_src = """
    a = 0.1;
    b1: sbend, l=2.0, angle:=a, k0:=0.2*a;
    seq: sequence, l=2.0;
    b1: b1, at=1;
    endsequence;
"""

mad = Madx()
mad.input(mad_src)
mad.beam()
mad.use('seq')

lmad = xt.Line.from_madx_sequence(mad.sequence.seq, deferred_expressions=True)
env = xt.load(string=mad_src, format='madx')
lenv = env['seq']

print(lenv.ref['b1'].edge_exit_angle_fdown._expr)
print(lmad.ref['b1'].edge_exit_angle_fdown._expr)
