import xtrack as xt
import xobjects as xo

src = """
    k1=0.2;

    q: quadrupole, l=1.0, k1=k1;

    seq: sequence, l=1;
    q1:q, at=0.5;
    endsequence;

    beam;
    use,sequence=seq;
    select,pattern=q1,flag=error;
    ealign, DPSI=0.3;
    """

from cpymad.madx import Madx
mad = Madx()
mad.input(src)

line = xt.Line.from_madx_sequence(mad.sequence.seq, apply_madx_errors=True)
line.set_particle_ref('positron', p0c=1e9)

twmad = mad.twiss(betx=1, bety=1, x=1e-3)
tw = line.twiss(betx=1, bety=1, x=1e-3)

xo.assert_allclose(line['q1'].rot_s_rad, 0.3, rtol=0, atol=1e-12)
xo.assert_allclose(tw.x[-1], twmad.x[-1], rtol=0, atol=1e-12)
xo.assert_allclose(tw.y[-1], twmad.y[-1], rtol=0, atol=1e-12)

