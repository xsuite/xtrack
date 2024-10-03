import numpy as np
import xtrack as xt
from xtrack.mad_parser.loader import MadxLoader
from cpymad.madx import Madx
import matplotlib.pyplot as plt

particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, p0c=7000e9)

mad = Madx(stdout=False)
mad.call('lhc.seq')
mad.call('squeeze_0.madx')
mad.input('beam, sequence=lhcb1, particle=proton, energy=7000;')
mad.use('lhcb1')
lhcb1_ref = xt.Line.from_madx_sequence(mad.sequence.lhcb1, deferred_expressions=True)
lhcb1_ref.particle_ref = particle_ref
tw_ref = lhcb1_ref.twiss4d()
tw_init = tw_ref.get_twiss_init('ip1')

loader = MadxLoader()
loader.load_file("lhc.seq")
loader.load_file("squeeze_0.madx")
env = loader.env
lhcb1 = env.lines['lhcb1']
lhcb1.particle_ref = particle_ref
tw = lhcb1.twiss4d()
# For debugging purposes, can try this:
# tw = lhcb1.twiss4d(start='ip1', end='_end_point', init=tw_init, _continue_if_lost=True)

plt.subplot(2, 1, 1)
plt.plot(tw_ref.s, tw_ref.betx, label='betx_ref')
plt.plot(tw.s, tw.betx, label='betx')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(tw_ref.s, tw_ref.bety, label='bety_ref')
plt.plot(tw.s, tw.bety, label='bety')
plt.legend()

plt.show()
