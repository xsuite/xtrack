from cpymad.madx import Madx

mad = Madx()
mad.call('lhc_sequence.madx')
mad.call('lhc_optics.madx')
mad.beam()
mad.sequence.lhcb1.use()

import xtrack as xt
import xpart as xp



from xtrack.mad_loader import MadLoader, MadElem

ml = MadLoader(mad.sequence.lhcb1)
ml.slicing_strategies=[(None,None,xt.mad_loader.UniformSlicing(10))]
line=ml.make_line()

line.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1, energy0=7e12)

line.build_tracker()
tw = line.twiss()
