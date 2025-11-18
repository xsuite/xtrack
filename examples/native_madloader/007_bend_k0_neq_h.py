import xtrack as xt
import xobjects as xo
from cpymad.madx import Madx

mad_src = """
    a = 0.1;
    b1: sbend, l=2.0, angle:=a, k0:=0.2*a, e1=0.02, e2=0.03, fint=1.5, hgap=0.04;
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

xo.assert_allclose(lenv['b1'].length,
                   lmad['b1'].length, rtol=0, atol=1e-12)
xo.assert_allclose(lenv['b1'].angle,
                   lmad['b1'].angle, rtol=0, atol=1e-12)
xo.assert_allclose(lenv['b1'].k0,
                   lmad['b1'].k0, rtol=0, atol=1e-12)
xo.assert_allclose(lenv['b1'].h,
                   lmad['b1'].h, rtol=0, atol=1e-12)
xo.assert_allclose(lenv['b1'].edge_entry_angle,
                   lmad['b1'].edge_entry_angle, rtol=0, atol=1e-12)
xo.assert_allclose(lenv['b1'].edge_exit_angle,
                   lmad['b1'].edge_exit_angle, rtol=0, atol=1e-12)
xo.assert_allclose(lenv['b1'].edge_entry_fint,
                   lmad['b1'].edge_entry_fint, rtol=0, atol=1e-12)
xo.assert_allclose(lenv['b1'].edge_exit_fint,
                     lmad['b1'].edge_exit_fint, rtol=0, atol=1e-12)
xo.assert_allclose(lenv['b1'].edge_entry_hgap,
                     lmad['b1'].edge_entry_hgap, rtol=0, atol=1e-12)
xo.assert_allclose(lenv['b1'].edge_exit_hgap,
                     lmad['b1'].edge_exit_hgap, rtol=0, atol=1e-12)
xo.assert_allclose(lenv['b1'].edge_entry_angle_fdown,
                   lmad['b1'].edge_entry_angle_fdown, rtol=0, atol=1e-12)
xo.assert_allclose(lenv['b1'].edge_exit_angle_fdown,
                   lmad['b1'].edge_exit_angle_fdown, rtol=0, atol=1e-12)