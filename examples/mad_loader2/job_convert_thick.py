from cpymad.madx import Madx

mad = Madx()
mad.call('lhc_sequence.madx')
mad.call('lhc_optics.madx')
mad.beam()
mad.sequence.lhcb1.use()

import xtrack as xt
import xpart as xp



from xtrack.mad_loader import MadLoader, MadElem, TeapotSlicing

ml = MadLoader(mad.sequence.lhcb1)

ml.slicing_strategies = [
    ml.make_slicing_strategy(
        name_regex=r'(mqt|mqtli|mqtlh)\..*',
        slicing_strategy=TeapotSlicing(2),
    ),
    ml.make_slicing_strategy(
        name_regex=r'(mbx|mbrb|mbrc|mbrs|mbh|mqwa|mqwb|mqy|mqm|mqmc|mqml)\..*',
        slicing_strategy=TeapotSlicing(4),
    ),
    ml.make_slicing_strategy(
        madx_type='mqxb',
        slicing_strategy=TeapotSlicing(16),
    ),
    ml.make_slicing_strategy(
        madx_type='mqxa',
        slicing_strategy=TeapotSlicing(16),
    ),
    ml.make_slicing_strategy(
        madx_type='mq',
        slicing_strategy=TeapotSlicing(2),
    ),
    ml.make_slicing_strategy(
        madx_type='mb',
        slicing_strategy=TeapotSlicing(2),
    ),
    ml.make_slicing_strategy(TeapotSlicing(1)),  # Default catch-all as in MAD-X
]
line=ml.make_line()

line.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1, energy0=7e12)

line.build_tracker()
tw = line.twiss(method='4d')
