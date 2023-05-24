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
    SlicingStrategy(name=r'(mqt|mqtli|mqtlh)\..*', slicing=TeapotSlicing(2)),
    SlicingStrategy(
        name=r'(mbx|mbrb|mbrc|mbrs|mbh|mqwa|mqwb|mqy|mqm|mqmc|mqml)\..*',
        slicing=TeapotSlicing(4),
    ),
    SlicingStrategy(madx_type='mqxb', slicing=TeapotSlicing(16)),
    SlicingStrategy(madx_type='mqxa', slicing=TeapotSlicing(16)),
    SlicingStrategy(madx_type='mq', slicing=TeapotSlicing(2)),
    SlicingStrategy(madx_type='mb', slicing=TeapotSlicing(2)),
    SlicingStrategy(TeapotSlicing(1)),  # Default catch-all as in MAD-X
]
line=ml.make_line()

line.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1, energy0=7e12)

line.build_tracker()
tw = line.twiss(method='4d')
