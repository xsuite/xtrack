import xtrack as xt
from cpymad.madx import Madx

mad = Madx()

mad.input("""
k1=0.2;
tilt=0.1;

elm: sbend,
    k1:=k1,
    l=1,
    angle=0.1,
    tilt=0.2;

seq: sequence, l=1;
elm: elm, at=0.5;
endsequence;

beam;
use, sequence=seq;
""")


lmad = xt.Line.from_madx_sequence(mad.sequence.seq)