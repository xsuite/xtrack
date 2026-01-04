import xtrack as xt
import xobjects as xo
import numpy as np

src = """

    xr: xrotation, angle=0.2;

    seq: sequence, l=1;
    xr, at=0.5;
    endsequence;

    beam;
    use,sequence=seq;
    """

from cpymad.madx import Madx
mad = Madx()
mad.input(src)

line = xt.Line.from_madx_sequence(mad.sequence.seq, apply_madx_errors=True)
line.set_particle_ref('positron', p0c=1e9)

twmad = mad.twiss(betx=1, bety=1, x=1e-3)
tw = line.twiss(betx=1, bety=1, x=1e-3)

xo.assert_allclose(line['xr'].angle, np.rad2deg(0.2), rtol=0, atol=1e-12)
xo.assert_allclose(tw.py[-1], np.sin(0.2), rtol=0, atol=1e-12)
xo.assert_allclose(twmad.py[-1], tw.py[-1], rtol=0, atol=1e-12)