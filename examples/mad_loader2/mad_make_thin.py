import xtrack as xt
import xpart as xp
from cpymad.madx import Madx
import matplotlib.pyplot as plt
from xtrack.mad_loader import MadLoader

mad = Madx()
mad.call('lhc_sequence.madx')
mad.call('lhc_optics.madx')
mad.call('slice.madx')
mad.beam()
mad.sequence.lhcb1.use()

ml = MadLoader(mad.sequence.lhcb1)
ml.slicing_strategies = []
line = ml.make_line()
line.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1, energy0=7e12)

line.build_tracker()
tw = line.twiss(method='4d')

plt.plot(tw.s, tw.x)
plt.show()
