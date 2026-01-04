import xtrack as xt
import xobjects as xo
import numpy as np

src = """

    tr: translation, dx=0.1, dy=0.2;

    seq: sequence, l=1;
    tr, at=0.5;
    endsequence;

    beam;
    use,sequence=seq;
    """

from cpymad.madx import Madx
mad = Madx()
mad.input(src)

line = xt.Line.from_madx_sequence(mad.sequence.seq, apply_madx_errors=True)
line.set_particle_ref('positron', p0c=1e9)

twmad = mad.twiss(betx=1, bety=1)
tw = line.twiss(betx=1, bety=1)

xo.assert_allclose(line['tr'].dx, 0.1, rtol=0, atol=1e-12)
xo.assert_allclose(line['tr'].dy, 0.2, rtol=0, atol=1e-12)
xo.assert_allclose(tw.x[-1], -0.1, rtol=0, atol=1e-12)
xo.assert_allclose(twmad.x[-1], tw.x[-1], rtol=0, atol=1e-12)
xo.assert_allclose(tw.y[-1], -0.2, rtol=0, atol=1e-12)
xo.assert_allclose(twmad.y[-1], tw.y[-1], rtol=0, atol=1e-12)
