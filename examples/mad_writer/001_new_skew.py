from cpymad.madx import Madx
import xtrack as xt

mad = Madx()
mad.input("""

a = 1.;
quad: quadrupole, k1:=a*3, l=1;
squad: quadrupole, k1s:=a*2, l=1;

testseq: sequence, l=10;
qn: quad, at=2., l=1;
qs: squad, at=5.;
m1: marker, at=6.;
endsequence;
beam;
use, sequence=testseq;
"""
)

line = xt.Line.from_madx_sequence(sequence=mad.sequence.testseq,
                                    deferred_expressions=True)
line.particle_ref = xt.Particles(p0c=7000e9, mass0=0.9382720813e9)

tw = line.twiss(betx=1, bety=2, start=line.element_names[0],
                                 end=line.element_names[-1])

twmad = mad.twiss(betx=1, bety=2, ripken=True)

print('betx2 mad:    ', twmad.beta12[-1])
print('betx2 xtrack: ', tw.betx2[-1])
print('bety2 mad:    ', twmad.beta22[-1])
print('bety2 xtrack: ', tw.bety2[-1])