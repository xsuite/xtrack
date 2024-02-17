from cpymad.madx import Madx
import xtrack as xt

# Things to check:
# - backtracking

mad = Madx()
mad.input("""


testseq: sequence, l=1;

squad1: quadrupole, k1s:=0.5, l=1, at=0.5;
! squad1: multipole, knl:={0, 0}, ksl:={0, 0.5}, at=1;

!quad: quadrupole, k1:=0.5, l=1, at=3;

!squad2: quadrupole, k1s:=-0.2, l=1, at=4;

endsequence;
beam;
use, sequence=testseq;
"""
)

line = xt.Line.from_madx_sequence(sequence=mad.sequence.testseq,
                                    deferred_expressions=True)
line.particle_ref = xt.Particles(p0c=7000e9, mass0=0.9382720813e9)

tw = line.twiss(x=1, betx=2, bety=1, start=line.element_names[0],
                                 end=line.element_names[-1])

twmad = mad.twiss(x=1, betx=2, bety=1, ripken=True)

print('betx2 mad:    ', twmad.beta12[-1])
print('betx2 xtrack: ', tw.betx2[-1])
print('bety1 mad:    ', twmad.beta21[-1])
print('bety1 xtrack: ', tw.bety1[-1])
print('x mad   :' , twmad.x[-1])
print('x xtrack:' , tw.x[-1])
print('y mad   :' , twmad.y[-1])
print('y xtrack:' , tw.y[-1])