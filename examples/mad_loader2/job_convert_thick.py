import xpart as xp
from cpymad.madx import Madx
from xtrack.mad_loader import MadLoader, TeapotSlicing, SlicingStrategy

mad = Madx()
mad.call('lhc_sequence.madx')
mad.call('lhc_optics.madx')
mad.beam()
mad.sequence.lhcb1.use()

ml = MadLoader(mad.sequence.lhcb1, enable_slicing=True)

ml.slicing_strategies = [
    SlicingStrategy(slicing=TeapotSlicing(1)),  # Default catch-all as in MAD-X
    SlicingStrategy(slicing=TeapotSlicing(2), madx_type='mb'),
    SlicingStrategy(slicing=TeapotSlicing(2), madx_type='mq'),
    SlicingStrategy(slicing=TeapotSlicing(16), madx_type='mqxa'),
    SlicingStrategy(slicing=TeapotSlicing(16), madx_type='mqxb'),
    SlicingStrategy(
        slicing=TeapotSlicing(4),
        name=r'(mbx|mbrb|mbrc|mbrs|mbh|mqwa|mqwb|mqy|mqm|mqmc|mqml)\..*',
    ),
    SlicingStrategy(slicing=TeapotSlicing(2), name=r'(mqt|mqtli|mqtlh)\..*'),
]
line = ml.make_line()

line.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1, energy0=7e12)

line.build_tracker()
tw = line.twiss(method='4d')
