import numpy as np
from cpymad.madx import Madx
import xtrack as xt

# We load the model from MAD-X lattice and strengths
env = xt.load('../../test_data/elena/elena.seq')
env.vars.load('../../test_data/elena/highenergy.str')
line = env.elena
line.set_particle_ref('antiproton', p0c=0.1e9)
line['beam_p_gev_c'] = line.particle_ref.p0c[0]/1e9 # Used by the optics

# Inspect one bend
line['lnr.mbhek.0135']
# returns:
#
# Bend(length=0.971, k0=1.08, k1=0, h=1.08, model='adaptive',
#      knl=array([0., 0., 0., 0., 0.]), ksl=array([0., 0., 0., 0., 0.]),
#      edge_entry_active=1, edge_exit_active=1,
#      edge_entry_model='linear', edge_exit_model='linear',
#      edge_entry_angle=0.287, edge_exit_angle=0.287,
#      edge_entry_angle_fdown=0, edge_exit_angle_fdown=0,
#      edge_entry_fint=0.424, edge_exit_fint=0.424,
#      edge_entry_hgap=0.038, edge_exit_hgap=0.038,
#      shift_x=0, shift_y=0, rot_s_rad=0)

# By default the adaptive model is used for the core and the linearized model for the edge
line['lnr.mbhek.0135'].model # is 'adaptive'
line['lnr.mbhek.0135'].edge_entry_model # is 'linear'
line['lnr.mbhek.0135'].edge_exit_model # is 'linear'

# For small machines (bends with large bending angles) it is more appropriate to
# switch to the `full` model for the edge
line.configure_bend_model(core='adaptive', edge='full')

# It is also possible to switch from the expanded drift to the exact one
line.configure_drift_model(model='exact')

line['lnr.mbhek.0135'].model # is 'adaptive'
line['lnr.mbhek.0135'].edge_entry_model # is 'full'
line['lnr.mbhek.0135'].edge_exit_model # is 'full'

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
line.configure_drift_model(model='expanded')

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
tt_sliced = line.get_table(attr=True)
tbends = tt_sliced.rows[tt_sliced.element_type == 'ThickSliceBend']
for ax in [ax1, ax2, ax3, ax4]:
    for nn in tbends.name:
        ax.axvspan(tbends['s', nn], tbends['s', nn] + tbends['length', nn],
                   color='b', alpha=0.2, linewidth=0)

plt.show()