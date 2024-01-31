import numpy as np
from cpymad.madx import Madx
import xtrack as xt

# We get the model from MAD-X
mad = Madx()
folder = ('../../test_data/elena')
mad.call(folder + '/elena.seq')
mad.call(folder + '/highenergy.str')
mad.call(folder + '/highenergy.beam')
mad.use('elena')

# Build xsuite line
seq = mad.sequence.elena
line = xt.Line.from_madx_sequence(seq)
line.particle_ref = xt.Particles(gamma0=seq.beam.gamma,
                                    mass0=seq.beam.mass * 1e9,
                                    q0=seq.beam.charge)

# Inspect one bend (the bend is a compound element made by a core and two edge elements)
tt = line.get_table()
tt.rows['lnr.mbhek.0135_entry':'lnr.mbhek.0135_exit'].show()
# name                          s element_type isthick ...
# lnr.mbhek.0135_entry     4.4992 Marker         False
# lnr.mbhek.0135_den       4.4992 DipoleEdge     False
# lnr.mbhek.0135           4.4992 Bend            True
# lnr.mbhek.0135_dex      5.46995 DipoleEdge     False
# lnr.mbhek.0135_exit     5.46995 Marker         False

# By default the adaptive model is used for the core and the linearized model for the edge
line['lnr.mbhek.0135'].model # is 'adaptive'
line['lnr.mbhek.0135_den'].model # is 'linear'

# For small machines (bends with large bending angles) it is more appropriate to
# switch to the `full` model for the edge
line.configure_bend_model(core='adaptive', edge='full')

# It is also possible to switch from the expanded drift to the exact one
line.config.XTRACK_USE_EXACT_DRIFTS = True

line['lnr.mbhek.0135'].model # is 'adaptive'
line['lnr.mbhek.0135_den'].model # is 'full'

# Slice the bends to see the behavior of the optics functions within them
line.slice_thick_elements(
    slicing_strategies=[
        xt.Strategy(slicing=None), # don't touch other elements
        xt.Strategy(slicing=xt.Uniform(10, mode='thick'), element_type=xt.Bend)
    ])

# Twiss
tw = line.twiss(method='4d')

# Switch to a simplified model
line.configure_bend_model(core='expanded', edge='linear')
line.config.XTRACK_USE_EXACT_DRIFTS = False

# Twiss with the default model
tw_simpl = line.twiss(method='4d')

# Compare beta functions and chromatic properties

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1, figsize=(6.4, 4.8 * 1.5))
ax1 = plt.subplot(4,1,1)
plt.plot(tw.s, tw.betx, label='adaptive')
plt.plot(tw_simpl.s, tw_simpl.betx, '--', label='simplified')
plt.ylabel(r'$\beta_x$')
plt.legend(loc='best')

ax2 = plt.subplot(4,1,2, sharex=ax1)
plt.plot(tw.s, tw.bety)
plt.plot(tw_simpl.s, tw_simpl.bety, '--')
plt.ylabel(r'$\beta_y$')

ax3 = plt.subplot(4,1,3, sharex=ax1)
plt.plot(tw.s, tw.wx_chrom)
plt.plot(tw_simpl.s, tw_simpl.wx_chrom, '--')
plt.ylabel(r'$W_x$')

ax4 = plt.subplot(4,1,4, sharex=ax1)
plt.plot(tw.s, tw.wy_chrom)
plt.plot(tw_simpl.s, tw_simpl.wy_chrom, '--')
plt.ylabel(r'$W_y$')
plt.xlabel('s [m]')

# Highlight the bends
tt_sliced = line.get_table()
tbends = tt_sliced.rows[tt_sliced.element_type == 'Bend']
for ax in [ax1, ax2, ax3, ax4]:
    for nn in tbends.name:
        ax.axvspan(tbends['s', nn], tbends['s', nn] + line[nn].length,
                   color='b', alpha=0.2, linewidth=0)

plt.show()