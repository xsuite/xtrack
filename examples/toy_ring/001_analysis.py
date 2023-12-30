import numpy as np
import xtrack as xt

# Save to json
line = xt.Line.from_json('toy_ring.json')
tw0 = line.twiss(method='4d')

xtsl = xt.slicing
# slicing to see beta inside bends
line_sliced = line.copy()
line_sliced.slice_thick_elements(
    slicing_strategies=[
        xtsl.Strategy(slicing=None), # default
        xtsl.Strategy(slicing=xtsl.Teapot(100, mode='thick'), element_type=xt.Bend),
        xtsl.Strategy(slicing=xtsl.Teapot(100, mode='thick'), element_type=xt.Quadrupole),
    ]
)

# tw = line.twiss(
#     start='mqf.1', end=len(line)-1, init=xt.TwissInit(betx=1, bety=1))
tw = line_sliced.twiss(method='4d')

tt = line.get_table(attr=True)

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1, figsize=(6.4, 4.8 * 1.4))

ax0 = plt.subplot(3, 1, 1)
ax0.bar(tt.s, tt.k1l, width=tt.length, color='r', alpha=0.2)
ax0.axhline(0, color='k', lw=0.5, alpha=0.5)
ax0.set_ylabel(r'k1L [m$^{-1}$]')
ax0.set_ylim(np.array([-1, 1]) * np.max(np.abs(tt.k1l)) * 1.1)

ax1 = plt.subplot(3, 1, 2, sharex=ax0)
plt.plot(tw.s, tw.betx, label='betx')
plt.plot(tw.s, tw.bety, label='bety')
# plt.plot(tw0.s, tw0.betx, '.', color='C0')
# plt.plot(tw0.s, tw0.bety, '.', color='C1')

plt.ylabel(r'$\beta_{x,y}$ [m]')
plt.legend()

ax2 = plt.subplot(3, 1, 3, sharex=ax1)
ax2.plot(tw.s, tw.dx, color='C2', label='dx')
ax2.set_ylabel(r'$D_x$ [m]')
plt.xlabel('s [m]')
plt.suptitle('Optics')

sv = line_sliced.survey()
plt.figure(2, figsize=(6.4 * 1.7, 4.8))

axsv1 = plt.subplot(1, 2, 1)
plt.plot(sv.s, sv.X, '-', label='X')
plt.plot(sv.s, sv.Z, '-', label='Z')
plt.xlabel('s [m]')
plt.ylabel('X, Z [m]')
plt.legend()
axsv2 = plt.subplot(1, 2, 2)
plt.plot(sv.Z, sv.X, '-')
plt.xlabel('Z [m]')
plt.ylabel('X [m]')
plt.axis('equal')
plt.suptitle('Survey')



tt = line.get_table(attr=True)
for ax in [ax1, ax2, axsv1]:
    ttbends = tt.rows['mb.*']
    for ii in range(len(ttbends)):
        ss = ttbends.s[ii]
        ll = ttbends.length[ii]
        ax.axvspan(ss, ss+ll, facecolor='b', alpha=0.2)

    ttquads = tt.rows['mq.*']
    for ii in range(len(ttquads)):
        ss = ttquads.s[ii]
        ll = ttquads.length[ii]
        ax.axvspan(ss, ss+ll, facecolor='r', alpha=0.2)


# Comparison of bend models
line_sliced.configure_bend_model(core='expanded', edge='linear')
tw_core_expand_fringe_linear = line_sliced.twiss(method='4d')
line_sliced.configure_bend_model(core='full', edge='linear')
tw_core_full_fringe_linear = line_sliced.twiss(method='4d')

plt.figure(3, figsize=(6.4, 4.8 * 1.4))
ax0 = plt.subplot(3, 1, 1)
ax0.plot(tw_core_expand_fringe_linear.s, tw_core_expand_fringe_linear.betx, label='betx')
ax0.plot(tw_core_full_fringe_linear.s, tw_core_full_fringe_linear.betx, label='betx')
ax0.set_ylabel(r'$\beta_{x}$ [m]')

# wx_chrom
ax1 = plt.subplot(3, 1, 2, sharex=ax0)
ax1.plot(tw_core_expand_fringe_linear.s, tw_core_expand_fringe_linear.wx_chrom, label='expanded')
ax1.plot(tw_core_full_fringe_linear.s, tw_core_full_fringe_linear.wx_chrom, label='full')
plt.ylabel(r'$W_x$')
plt.legend()

# wy_chrom
ax2 = plt.subplot(3, 1, 3, sharex=ax0)
ax2.plot(tw_core_expand_fringe_linear.s, tw_core_expand_fringe_linear.wy_chrom, label='expanded')
ax2.plot(tw_core_full_fringe_linear.s, tw_core_full_fringe_linear.wy_chrom, label='full')
plt.ylabel(r'$W_y$')


plt.show()