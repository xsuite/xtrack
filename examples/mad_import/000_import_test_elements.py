from scipy.constants import c as clight
import numpy as np

from cpymad.madx import Madx
import xtrack as xt
import xpart as xp

mad = Madx()

# Element definitions
mad.input("""
cav0: rfcavity, freq=10, lag=0.5, volt=6;
cav1: rfcavity, lag=0.5, volt=6, harmon=8;
wire1: wire, current=5, l=0, l_phy=1, l_int=2, xma=1e-3, yma=2e-3;
""")

# Sequence
mad.input("""

testseq: sequence, l=10;
c0: cav0, at=0.2, apertype=circle, aperture=0.01;
c1: cav1, at=0.2, apertype=circle, aperture=0.01;

w: wire1, at=1;

endsequence;
"""
)

# Beam
mad.input("""
beam, particle=proton, gamma=1.05, sequence=testseq;
""")


mad.use('testseq')

seq = mad.sequence['testseq']
line = xt.Line.from_madx_sequence(sequence=seq)
line.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, gamma0=1.05)


assert len(line.element_names) == len(line.element_dict.keys())
assert line.get_length() == 10

assert isinstance(line['c0'], xt.Cavity)
assert line.get_s_position('c0') == 0.2
assert line['c0'].frequency == 10e6
assert line['c0'].lag == 180
assert line['c0'].voltage == 6e6

assert isinstance(line['c1'], xt.Cavity)
assert line.get_s_position('c1') == 0.2
assert np.isclose(line['c1'].frequency, clight*line.particle_ref.beta0/10.*8,
                  rtol=0, atol=1e-7)
assert line['c1'].lag == 180
assert line['c1'].voltage == 6e6

assert isinstance(line['w'], xt.Wire)
assert line.get_s_position('w') == 1
assert line['w'].wire_L_phy == 1
assert line['w'].wire_L_int == 2
assert line['w'].wire_xma == 1e-3
assert line['w'].wire_yma == 2e-3



