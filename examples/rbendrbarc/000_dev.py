import numpy as np
import xtrack as xt
import xobjects as xo
from cpymad.madx import Madx

mad = Madx()
mad.input('''
    ang = 0.6;
    lb = 0.5;
    rb: rbend, l=lb, angle=ang;

    beam, particle=proton, energy=1.0;
    seq: sequence, l=2;
    rb, at=1;
    endsequence;
    use, sequence=seq;
''')
tw = xt.Table(mad.twiss(betx=1.0, bety=1.0))

ds_madx = tw['s', 'rb:1'] - tw['s', 'rb:1<<1']

line = xt.Line.from_madx_sequence(mad.sequence.seq)

xo.assert_allclose(ds_madx, line['rb'].length, atol=0, rtol=1e-12)

env = xt.Environment()
env['lb'] = 0.5
env['ang'] = 0.6
env.new('rb_rbarc', xt.Bend)
env.set('rb_rbarc', length='lb', angle='ang', rbend=True, rbarc=True)

xo.assert_allclose(env['rb_rbarc'].length, ds_madx, atol=0, rtol=1e-12)
xo.assert_allclose(env['rb_rbarc'].h * env['rb_rbarc'].length, 0.6, atol=0, rtol=1e-12)
xo.assert_allclose(env['rb_rbarc'].edge_entry_angle, 0.3, atol=0, rtol=1e-12)
xo.assert_allclose(env['rb_rbarc'].edge_exit_angle, 0.3, atol=0, rtol=1e-12)