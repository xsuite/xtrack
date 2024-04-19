import numpy as np
from cpymad.madx import Madx
import xtrack as xt

folder = ('../../test_data/elena')
mad = Madx()

mad.call(folder + '/elena.seq')

mad.call(folder + '/highenergy.str')
mad.call(folder + '/highenergy.beam')

mad.use('elena')

seq = mad.sequence.elena
line = xt.Line.from_madx_sequence(seq)
line.particle_ref = xt.Particles(gamma0=seq.beam.gamma,
                                    mass0=seq.beam.mass * 1e9,
                                    q0=seq.beam.charge)
# tt = line.get_table()
# tquads = tt.rows[tt.element_type == 'Quadrupole']
# for nn in tquads.name:
#     line[nn].k1 = 0
#     mad.input(f'{nn}, k1 = 0;')


line.slice_thick_elements(
        slicing_strategies=[
            xt.Strategy(slicing=None),
            xt.Strategy(slicing=xt.Uniform(10, mode='thick'), element_type=xt.Bend)
        ])

line.configure_bend_model(core='expanded', edge='linear') # default
tw_el = line.twiss(method='4d')

line.configure_bend_model(core='full', edge='linear')
tw_fl = line.twiss(method='4d')

line.configure_bend_model(core='expanded', edge='full')
tw_ef = line.twiss(method='4d')

line.configure_bend_model(core='full', edge='full')
tw_ff = line.twiss(method='4d')

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1, figsize=(4.8*1.8, 4.8))
ax01 = plt.subplot(1, 1, 1)
plt.plot(tw_el.s, tw_el.wx_chrom, label='core: expanded, edge: linear')
plt.plot(tw_fl.s, tw_fl.wx_chrom, label='core: full, edge: linear')
plt.plot(tw_ef.s, tw_ef.wx_chrom, label='core: expanded, edge: full')
plt.plot(tw_ff.s, tw_ff.wx_chrom, label='core: full, edge: full', lw=2, color='k')

plt.ylim(0, 14)
plt.legend(loc='best')
plt.xlabel('s [m]')
plt.ylabel(r'$W_x$')

plt.figure(11, figsize=(4.8*1.8, 4.8))
ax11 = plt.subplot(1, 1, 1)
plt.plot(tw_el.s, tw_el.wy_chrom, label='core: expanded, edge: linear')
plt.plot(tw_fl.s, tw_fl.wy_chrom, label='core: full, edge: linear')
plt.plot(tw_ef.s, tw_ef.wy_chrom, label='core: expanded, edge: full')
plt.plot(tw_ff.s, tw_ff.wy_chrom, label='core: full, edge: full', lw=2, color='k')

plt.legend(loc='best')
plt.xlabel('s [m]')
plt.ylabel(r'$W_y$')


# Same for betx
plt.figure(2, figsize=(4.8*1.8, 4.8))
ax2 = plt.subplot(1, 1, 1)
plt.plot(tw_el.s, tw_el.betx, label='core: expanded, edge: linear')
plt.plot(tw_fl.s, tw_fl.betx, label='core: full, edge: linear')
plt.plot(tw_ef.s, tw_ef.betx, label='core: expanded, edge: full')
plt.plot(tw_ff.s, tw_ff.betx, label='core: full, edge: full', lw=2, color='k')

plt.ylim(0, 14)
plt.legend(loc='best')
plt.xlabel('s [m]')
plt.ylabel(r'$\beta_x$ [m]')

tw= line.twiss(method='4d')
plt.figure(3, figsize=(4.8*1.8, 4.8))
ax3 = plt.subplot(1, 1, 1)
plt.plot(tw.s, tw.betx, label='betx')
plt.plot(tw.s, tw.bety, label='bety')
plt.legend(loc='best')
plt.xlabel('s [m]')
plt.ylabel(r'$\beta_{x,y}$ [m]')

plt.figure(4, figsize=(4.8*1.8, 4.8))
ax4 = plt.subplot(1, 1, 1)
plt.plot(tw.s, tw.wx_chrom, label='wx')
plt.plot(tw.s, tw.wy_chrom, label='wy')
plt.legend(loc='best')
plt.xlabel('s')
plt.ylabel(r'$W_{x,y}$')


for ax in [ax01, ax2, ax3, ax4, ax11]:
    tt = line.get_table()
    tbends = tt.rows[tt.element_type == 'Bend']
    tquads = tt.rows[tt.element_type == 'Quadrupole']
    for nn in tbends.name:
        ax.axvspan(tbends['s', nn], tbends['s', nn] + line[nn].length, color='b',
                    alpha=0.2, lw=0)
    for nn in tquads.name:
        ax.axvspan(tquads['s', nn], tquads['s', nn] + line[nn].length, color='r',
                    alpha=0.2, lw=0)

plt.show()