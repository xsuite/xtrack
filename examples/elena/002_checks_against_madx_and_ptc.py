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

mad.input('twiss, chrom, table=twchr;')
mad.input('twiss, chrom=false, table=twnochr;')


line.slice_thick_elements(
        slicing_strategies=[
            xt.Strategy(slicing=None),
            xt.Strategy(slicing=xt.Uniform(10, mode='thick'), element_type=xt.Bend)
        ])

line.configure_bend_model(core='expanded', edge='linear') # default
tw_el = line.twiss(method='4d', group_compound_elements=True)

line.configure_bend_model(core='full', edge='linear')
tw_fl = line.twiss(method='4d', group_compound_elements=True)

line.configure_bend_model(core='expanded', edge='full')
tw_ef = line.twiss(method='4d', group_compound_elements=True)

line.configure_bend_model(core='full', edge='full')
tw_ff = line.twiss(method='4d', group_compound_elements=True)

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1, figsize=(4.8*1.8, 4.8))
ax1 = plt.subplot(1, 1, 1)
plt.plot(tw_el.s, tw_el.wx_chrom, label='core: expanded, edge: linear')
plt.plot(tw_fl.s, tw_fl.wx_chrom, label='core: full, edge: linear')
plt.plot(tw_ef.s, tw_ef.wx_chrom, label='core: expanded, edge: full')
plt.plot(tw_ff.s, tw_ff.wx_chrom, label='core: full, edge: full', lw=2, color='k')

twmad = mad.table.twchr
plt.plot(twmad.s, twmad.wx*tw_ff.beta0, '--', label='madx', color='r', alpha=0.9)

plt.ylim(0, 14)
plt.legend(loc='best')
plt.xlabel('s [m]')
plt.ylabel(r'$W_x$')

plt.figure(11, figsize=(4.8*1.8, 4.8))
ax1 = plt.subplot(1, 1, 1)
plt.plot(tw_el.s, tw_el.wy_chrom, label='core: expanded, edge: linear')
plt.plot(tw_fl.s, tw_fl.wy_chrom, label='core: full, edge: linear')
plt.plot(tw_ef.s, tw_ef.wy_chrom, label='core: expanded, edge: full')
plt.plot(tw_ff.s, tw_ff.wy_chrom, label='core: full, edge: full', lw=2, color='k')

twmad = mad.table.twchr
plt.plot(twmad.s, twmad.wy*tw_ff.beta0, '--', label='madx', color='r', alpha=0.9)

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
plt.plot(twmad.s, twmad.betx, '--', label='madx', color='r', alpha=0.9)

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


for ax in [ax1, ax2, ax3, ax4]:
    tt = line.get_table()
    tbends = tt.rows[tt.element_type == 'Bend']
    tquads = tt.rows[tt.element_type == 'Quadrupole']
    for nn in tbends.name:
        ax.axvspan(tbends['s', nn], tbends['s', nn] + line[nn].length, color='b',
                    alpha=0.2, lw=0)
    for nn in tquads.name:
        ax.axvspan(tquads['s', nn], tquads['s', nn] + line[nn].length, color='r',
                    alpha=0.2, lw=0)


mad.input('twiss, chrom=false, table=twnochr;')
twmad_nc = mad.table.twnochr

delta_chrom = 1e-4
mad.input(f'''
  ptc_create_universe;
  !ptc_create_layout, time=false, model=2, exact=true, method=6, nst=10;
  ptc_create_layout, time=false, model=1, exact=true, method=6, nst=10;
    select, flag=ptc_twiss, clear;
    select, flag=ptc_twiss, column=name,keyword,s,l,x,px,y,py,beta11,beta22,disp1,k1l;
    ptc_twiss, closed_orbit, icase=56, no=2, deltap=0, table=ptc_twiss,
               summary_table=ptc_twiss_summary, slice_magnets=true;
    ptc_twiss, closed_orbit, icase=56, no=2, deltap={delta_chrom:e}, table=ptc_twiss_pdp,
               summary_table=ptc_twiss_summary_pdp, slice_magnets=true;
    ptc_twiss, closed_orbit, icase=56, no=2, deltap={-delta_chrom:e}, table=ptc_twiss_mdp,
               summary_table=ptc_twiss_summary_mdp, slice_magnets=true;

    #### TEEEEST
    ptc_twiss, closed_orbit, icase=56, no=2, deltap={delta_chrom:e}, table=ptc_twiss_pdp_o,
               alfx=0, betx=1, alfy=0, bety=1, betz=1, dx=0, dy=0, dpx=0, dpy=0,
               summary_table=ptc_twiss_summary_pdp_o, slice_magnets=true;
  ptc_end;
''')

qx_ptc = mad.table.ptc_twiss.mu1[-1]
qy_ptc = mad.table.ptc_twiss.mu2[-1]
dq1_ptc = (mad.table.ptc_twiss_pdp.mu1[-1] - mad.table.ptc_twiss_mdp.mu1[-1]) / (2 * delta_chrom)
dq2_ptc = (mad.table.ptc_twiss_pdp.mu2[-1] - mad.table.ptc_twiss_mdp.mu2[-1]) / (2 * delta_chrom)

tptc = mad.table.ptc_twiss
tptc_p = mad.table.ptc_twiss_pdp
tptc_m = mad.table.ptc_twiss_mdp

fp = 1 + delta_chrom
fm = 1 - delta_chrom

betx = 0.5 * (tptc_p.beta11 / fp + tptc_m.beta11 / fm)
alfx = 0.5 * (tptc_p.alfa11 + tptc_m.alfa11)
d_betx = (tptc_p.beta11 / fp - tptc_m.beta11 / fm)/ (2 * delta_chrom)
d_alfx = (tptc_p.alfa11 - tptc_m.alfa11)/ (2 * delta_chrom)

bx_ptc = d_betx / betx
ax_ptc = d_alfx - d_betx * alfx / betx
wx_ptc = np.sqrt(ax_ptc**2 + bx_ptc**2)

bety = 0.5 * (tptc_p.beta22 / fp  + tptc_m.beta22 / fm)
alfy = 0.5 * (tptc_p.alfa22 + tptc_m.alfa22)
d_bety = (tptc_p.beta22 / fp - tptc_m.beta22 / fm)/ (2 * delta_chrom)
d_alfy = (tptc_p.alfa22 - tptc_m.alfa22)/ (2 * delta_chrom)

by_ptc = d_bety / bety
ay_ptc = d_alfy - alfy * by_ptc
wy_ptc = np.sqrt(ay_ptc**2 + by_ptc**2)


print(f'qx xsuite:          {tw_ff.qx}')
print(f'qx ptc:             {qx_ptc}')
print(f'qx mad (chrom=F):   {twmad_nc.summary.q1}')
print(f'qy xsuite:          {tw_ff.qy}')
print(f'qy ptc:             {qy_ptc}')
print(f'qy mad (chrom=F):   {twmad_nc.summary.q2}')
print(f'dqx xsuite:         {tw_ff.dqx}')
print(f'dqx ptc:            {dq1_ptc}')
print(f'dqx mad (chrom=F):  {twmad_nc.summary.dq1 * seq.beam.beta}')
print(f'dqy xsuite:         {tw_ff.dqy}')
print(f'dqy ptc:            {dq2_ptc}')
print(f'dqy mad (chrom=F):  {twmad_nc.summary.dq2 * seq.beam.beta}')


plt.figure(101, figsize=(6.4, 4.8 * 1.5))

ax1 = plt.subplot(4,1,1)
plt.plot(tptc.s, ax_ptc, label='ptc')
plt.plot(tw.s, tw.ax_chrom, label='xsuite')
plt.ylabel(r'$A_x$')
plt.legend(loc='best')

ax2 = plt.subplot(4,1,2, sharex=ax1)
plt.plot(tptc.s, bx_ptc)
plt.plot(tw.s, tw.bx_chrom)
plt.ylabel(r'$B_x$')

ax3 = plt.subplot(4,1,3, sharex=ax1)
plt.plot(tptc.s, ay_ptc)
plt.plot(tw.s, tw.ay_chrom)
plt.ylabel(r'$A_y$')

ax4 = plt.subplot(4,1,4, sharex=ax1)
plt.plot(tptc.s, by_ptc)
plt.plot(tw.s, tw.by_chrom)
plt.ylabel(r'$B_y$')
plt.xlabel('s [m]')

for ax in [ax1, ax2, ax3, ax4]:
    tt = line.get_table()
    tbends = tt.rows[tt.element_type == 'Bend']
    tquads = tt.rows[tt.element_type == 'Quadrupole']
    for nn in tbends.name:
        ax.axvspan(tbends['s', nn], tbends['s', nn] + line[nn].length, color='b',
                    alpha=0.2, lw=0)
    for nn in tquads.name:
        ax.axvspan(tquads['s', nn], tquads['s', nn] + line[nn].length, color='r',
                    alpha=0.2, lw=0)

two = line.twiss(method='4d', start=line.element_names[0], end='_end_point',
                 betx=tptc.beta11[0], bety=tptc.beta22[0],
                 alfx=tptc.alfa11[0], alfy=tptc.alfa22[0],
                 dx=tptc.dx[0], dy=tptc.dy[0],
                 dpx=tptc.dpx[0], dpy=tptc.dpy[0],
                 ax_chrom=ax_ptc[0], bx_chrom=bx_ptc[0],
                 ay_chrom=ay_ptc[0], by_chrom=by_ptc[0],
                 compute_chromatic_properties=True)

twp = line.twiss(method='4d', delta0=delta_chrom)

# Same for beta and orbit
plt.figure(102, figsize=(6.4, 4.8 * 1.5))

ax1 = plt.subplot(4,1,1)
plt.plot(tptc.s, tptc.beta11, label='ptc')
plt.plot(tw.s, tw.betx, label='xsuite')
plt.ylabel(r'$\beta_x$')
plt.legend(loc='best')

plt.subplot(4,1,2, sharex=ax1)
plt.plot(tptc.s, tptc.beta22)
plt.plot(tw.s, tw.bety)
plt.ylabel(r'$\beta_y$')

plt.subplot(4,1,3, sharex=ax1)
plt.plot(tptc.s, tptc.x)
plt.plot(tw.s, tw.x)
plt.ylabel(r'$x$')

plt.subplot(4,1,4, sharex=ax1)
plt.plot(tptc.s, tptc.y)
plt.plot(tw.s, tw.y)
plt.ylabel(r'$y$')
plt.xlabel('s [m]')

plt.show()