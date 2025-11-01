import xtrack as xt
import xobjects as xo

mad_src = '''
    m1: multipole, knl={0, 0.01, 0.0, 0,0, 0.1, 0.3, 0.7};

    seq: sequence, l=10.0;
    m1a: m1, at=2.0;
    endsequence;
'''

env = xt.load(string=mad_src, format='madx')
xo.assert_allclose(env.elements['m1'].knl,
                   [0, 0.01, 0.0, 0, 0, 0.1, 0.3, 0.7], rtol=0, atol=1e-12)

from cpymad.madx import Madx
mad = Madx()
mad.input(mad_src)
mad.beam()
mad.use('seq')
line = xt.Line.from_madx_sequence(mad.sequence.seq)
xo.assert_allclose(line['m1a'].knl,
                   [0, 0.01, 0.0, 0, 0, 0.1, 0.3, 0.7], rtol=0, atol=1e-12)
