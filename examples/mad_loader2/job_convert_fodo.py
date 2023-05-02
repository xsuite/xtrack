from cpymad.madx import Madx

mad = Madx()
mad.input("""
q1: quadrupole,l=0.1,k1=8;
q2: quadrupole,l=0.1,k1=-10;

seq: sequence,l=3.1;
q1, at=2;
q2, at=3;
endsequence;

beam;

use,sequence=seq;
twiss;
""")

import xtrack as xt
import xpart as xp

from xtrack.mad_loader import MadLoader, MadElem

ml = MadLoader(mad.sequence.seq)
ml.slicing_strategies=[(None,None,xt.mad_loader.UniformSlicing(4))]
line=ml.make_line()

line.particle_ref = xp.Particles(
                    mass0=xp.PROTON_MASS_EV, q0=1, energy0=7e12)

line.build_tracker()
tw = line.twiss()
