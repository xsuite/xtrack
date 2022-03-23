from cpymad.madx import Madx
import xtrack as xt

mad = Madx()

# Element definitions
mad.input("""
cav0: rfcavity, freq=10, lag=0.5, volt=6;
wire1: wire, current=5, l=0, l_phy=1, l_int=2, xma=1e-3, yma=2e-3;
""")

# Sequence
mad.input("""

testseq: sequence, l=10;
c0: cav0, at=0., apertype=circle, aperture=0.01;
w: wire1, at=1;

endsequence;
"""
)

seq = mad.sequence['testseq']
line = xt.Line.from_madx_sequence(sequence=seq)

assert line['c0'].frequency == 10e6
assert line['c0'].lag == 180
assert line['c0'].voltage == 6e6


