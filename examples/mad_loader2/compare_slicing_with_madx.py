import xtrack as xt
import xpart as xp
from cpymad.madx import Madx
from xtrack.mad_loader import MadLoader, SlicingStrategy, TeapotSlicing
import numpy as np
import matplotlib.pyplot as plt
import re


# Make thin using madx
mad1 = Madx()
mad1.call('lhc_sequence.madx')
mad1.call('lhc_optics.madx')
mad1.call('slice.madx')
mad1.beam()
mad1.sequence.lhcb1.use()

line_mad = xt.Line.from_madx_sequence(mad1.sequence.lhcb1)
line_mad.cycle(name_first_element='ip3')
line_mad.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1, energy0=7e12)

line_mad.build_tracker()
line_mad.optimize_for_tracking()
tw_mad = line_mad.twiss(method='4d', strengths=True)

# Do the same in xsuite
mad2 = Madx()
mad2.call('lhc_sequence.madx')
mad2.call('lhc_optics.madx')
mad2.beam()
mad2.sequence.lhcb1.use()

ml = MadLoader(mad2.sequence.lhcb1, enable_slicing=True)

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
line_xt = ml.make_line()
line_xt.cycle(name_first_element='ip3')
line_xt.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1, energy0=7e12)

line_xt.build_tracker()
line_xt.optimize_for_tracking()
tw_xt = line_xt.twiss(method='4d', strengths=True)

# Compare
di_mad = tw_mad.rows[np.abs(tw_mad.k0nl) > 0]
di_xt = tw_xt.rows[np.abs(tw_xt.k0nl) > 0]

assert np.sum(di_mad.k0nl) == np.sum(di_xt.k0nl)

print(f"xtrack: qx = {tw_xt.qx}, qx = {tw_xt.qx}")
print(f"madx:   qx = {tw_mad.qx}, qx = {tw_mad.qx}")


def plot_coord_for_elements(coord):
    fig, ax = plt.subplots()
    ax.plot(tw_xt.s, tw_xt[coord], '.-', color='r')
    ax.plot(tw_mad.s, tw_mad[coord], '.-', color='b')

    for i, txt in enumerate(tw_xt.name):
        if txt.startswith('drift'):
            continue
        ax.annotate(txt, (tw_xt.s[i], tw_xt[coord][i]), color='r')

    for i, txt in enumerate(tw_mad.name):
        if txt.startswith('drift'):
            continue
        ax.annotate(txt, (tw_mad.s[i], tw_mad[coord][i]), color='b')

    fig.show()


# Compare x along the lines thinned in madx and xsuite:
# plot_coord_for_elements('x')
