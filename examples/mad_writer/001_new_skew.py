from cpymad.madx import Madx
import xtrack as xt

mad = Madx()
mad.input("""

a = 1.;


testseq: sequence, l=10;

squad1: quadrupole, k1s:=a*0.5, l=1, at=1;

quad: quadrupole, k1s:=1, l=1, at=3;

squad2: quadrupole, k1s:=-a*0.2, l=1, at=4;

m1: marker, at=10.;
endsequence;
beam;
use, sequence=testseq;
"""
)

line = xt.Line.from_madx_sequence(sequence=mad.sequence.testseq,
                                    deferred_expressions=True)
line.particle_ref = xt.Particles(p0c=7000e9, mass0=0.9382720813e9)

tw = line.twiss(betx=1, bety=1, start=line.element_names[0],
                                 end=line.element_names[-1])

twmad = mad.twiss(betx=1, bety=1, ripken=True)

print('betx2 mad:    ', twmad.beta12[-1])
print('betx2 xtrack: ', tw.betx2[-1])
print('bety2 mad:    ', twmad.beta21[-1])
print('bety2 xtrack: ', tw.bety1[-1])