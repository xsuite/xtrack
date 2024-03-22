import xtrack as xt
from cpymad.madx import Madx

import numpy as np

mad = Madx()

tilt_deg = 12
tilt_rad = np.deg2rad(tilt_deg)

mad.input(f"""
k1=0.2;

elm: quadrupole,
    k1:=k1,
    l=1,
    tilt={tilt_rad};

seq: sequence, l=1;
elm: elm, at=0.5;
endsequence;

beam;
use, sequence=seq;
""")


lmad = xt.Line.from_madx_sequence(mad.sequence.seq)
lmad.build_tracker()

bend = lmad['elm'].copy()

bend.tilt = tilt_deg

p0 = xt.Particles(x=1e-3, px=2e-3, y=3e-3, py=4e-3, p0c=1e9)

pmad = p0.copy()
ptest = p0.copy()

bend.track(ptest)
lmad.track(pmad)

pmad.get_table().show()
ptest.get_table().show()