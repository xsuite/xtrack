import xtrack as xt
import xpart as xp
import numpy as np
import matplotlib.pyplot as plt

from cpymad.madx import Madx
from xtrack.slicing import Strategy, Teapot


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
# line_mad.optimize_for_tracking()
tw_mad = line_mad.twiss(method='4d', strengths=True)

# Do the same in xsuite
mad2 = Madx()
mad2.call('lhc_sequence.madx')
mad2.call('lhc_optics.madx')
mad2.beam()
mad2.sequence.lhcb1.use()

thick_line = xt.Line.from_madx_sequence(mad2.sequence.lhcb1, allow_thick=True)
thick_line.cycle(name_first_element='ip3')

thick_line.build_tracker()
thick_line.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1, energy0=7e12)
tw_thick = thick_line.twiss(method='4d', strengths=True)

slicing_strategies = [
    Strategy(slicing=Teapot(1)),  # Default catch-all as in MAD-X
    Strategy(slicing=Teapot(2), name=r'^(mb|mq).*'),
    Strategy(slicing=Teapot(16), name=r'^(mqxa|mqxb).*'),
    Strategy(
        slicing=Teapot(4),
        name=r'(mbx|mbrb|mbrc|mbrs|mbh|mqwa|mqwb|mqy|mqm|mqmc|mqml)\..*',
    ),
    Strategy(slicing=Teapot(2), name=r'(mqt|mqtli|mqtlh)\..*'),
]

thin_line = thick_line.make_thin_line(slicing_strategies)
thin_line.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1, energy0=7e12)

thin_line.build_tracker()
# thin_line.optimize_for_tracking()
tw_thin = thin_line.twiss(method='4d', strengths=True)

# Compare
di_mad = tw_thin.rows[np.abs(tw_thin.k0nl) > 0]
di_xt = tw_thin.rows[np.abs(tw_thin.k0nl) > 0]

assert np.sum(di_mad.k0nl) == np.sum(di_xt.k0nl)

print(f"xtrack thin:  qx = {tw_thin.qx}, qx = {tw_thin.qx}")
print(f"madx thin:    qx = {tw_mad.qx}, qx = {tw_mad.qx}")
print(f"xtrack thick: qx = {tw_thick.qx}, qx = {tw_thick.qx}")


def plot_coord_for_elements(coord):
    fig, ax = plt.subplots()
    ax.plot(tw_thin.s, tw_thin[coord], '.-', color='r')
    ax.plot(tw_thick.s, tw_thick[coord], '.-', color='b')

    for i, txt in enumerate(tw_thin.name):
        if txt.startswith('drift'):
            continue
        ax.annotate(txt, (tw_thin.s[i], tw_thin[coord][i]), color='r')

    for i, txt in enumerate(tw_thin.name):
        if txt.startswith('drift'):
            continue
        ax.annotate(txt, (tw_thin.s[i], tw_thin[coord][i]), color='b')

    fig.show()


# Compare x along the lines thinned in madx and xsuite:
# plot_coord_for_elements('x')