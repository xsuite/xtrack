import xtrack as xt
from cpymad.madx import Madx

import numpy as np

mad = Madx()

mad.input("""
k1=0.2;

elm: quadrupole,
    k1:=k1,
    l=1,
    tilt=0.2;

seq: sequence, l=1;
elm: elm, at=0.5;
endsequence;

beam;
use, sequence=seq;
""")


lmad = xt.Line.from_madx_sequence(mad.sequence.seq)
lmad.build_tracker()

bend = lmad['elm'].copy()

bend._sin_tilt = np.sin(0.2)
bend._cos_tilt = np.cos(0.2)

p0 = xt.Particles(x=1e-3, px=2e-3, y=3e-3, py=4e-3, p0c=1e9)

pmad = p0.copy()
ptest = p0.copy()

bend.track(ptest)
lmad.track(pmad)

pmad.get_table()
ptest.get_table()