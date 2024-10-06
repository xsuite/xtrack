import numpy as np
import xtrack as xt
import xobjects as xo
from cpymad.madx import Madx

mad = Madx()
mad.input('''
    rb: rbend, l=1.0, angle=0.5;

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
env.new('rbe', xt.Bend)
env.set('rbe', length=1.0, angle=0.5, rbend=True, rbarc=True)

xo.assert_allclose(ds_madx, env['rbe'].length, atol=0, rtol=1e-12)
