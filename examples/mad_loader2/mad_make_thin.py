import xtrack as xt
import xpart as xp
from cpymad.madx import Madx

mad = Madx()
mad.call('lhc_sequence.madx')
mad.call('lhc_optics.madx')
mad.call('slice.madx')
mad.beam()
mad.sequence.lhcb1.use()

line = xt.Line.from_madx_sequence(mad.sequence.lhcb1)
line.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1, energy0=7e12)

line.build_tracker()
tw = line.twiss(method='4d')

