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

line.configure_bend_model(core='full', edge='full')
tw = line.twiss(method='4d')


twmad = mad.table.twchr
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

# The MAD-X PTC interface rescales the beta functions with (1 + deltap)
# see: https://github.com/MethodicalAcceleratorDesign/MAD-X/blob/eb495b4f926db53f3cd05133638860f910f42fe2/src/madx_ptc_twiss.f90#L1982
# We need to undo that
beta11_p = tptc_p.beta11 / fp
beta11_m = tptc_m.beta11 / fm
beta22_p = tptc_p.beta22 / fp
beta22_m = tptc_m.beta22 / fm
alfa11_p = tptc_p.alfa11
alfa11_m = tptc_m.alfa11
alfa22_p = tptc_p.alfa22
alfa22_m = tptc_m.alfa22

betx = 0.5 * (beta11_p + beta11_m)
bety = 0.5 * (beta22_p + beta22_m)
alfx = 0.5 * (alfa11_p + alfa11_m)
alfy = 0.5 * (alfa22_p + alfa22_m)
d_betx = (beta11_p - beta11_m) / (2 * delta_chrom)
d_bety = (beta22_p - beta22_m) / (2 * delta_chrom)
d_alfx = (alfa11_p - alfa11_m) / (2 * delta_chrom)
d_alfy = (alfa22_p - alfa22_m) / (2 * delta_chrom)

bx_ptc = d_betx / betx
by_ptc = d_bety / bety
ax_ptc = d_alfx - d_betx * alfx / betx
ay_ptc = d_alfy - d_bety * alfy / bety
wx_ptc = np.sqrt(ax_ptc**2 + bx_ptc**2)
wy_ptc = np.sqrt(ay_ptc**2 + by_ptc**2)

print(f'qx xsuite:          {tw.qx}')
print(f'qx ptc:             {qx_ptc}')
print(f'qx mad (chrom=F):   {twmad_nc.summary.q1}')
print(f'qy xsuite:          {tw.qy}')
print(f'qy ptc:             {qy_ptc}')
print(f'qy mad (chrom=F):   {twmad_nc.summary.q2}')
print(f'dqx xsuite:         {tw.dqx}')
print(f'dqx ptc:            {dq1_ptc}')
print(f'dqx mad (chrom=F):  {twmad_nc.summary.dq1 * seq.beam.beta}')
print(f'dqy xsuite:         {tw.dqy}')
print(f'dqy ptc:            {dq2_ptc}')
print(f'dqy mad (chrom=F):  {twmad_nc.summary.dq2 * seq.beam.beta}')


import matplotlib.pyplot as plt
plt.close('all')
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