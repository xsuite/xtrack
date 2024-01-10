from cpymad.madx import Madx

folder = '../../test_data/leir'
mad = Madx()

mad.call(folder + '/leir_inj_nominal.beam')
mad.input('BRHO = BEAM->PC * 3.3356 * (208./54.)')
mad.call(folder + '/leir.seq')
mad.call(folder + '/leir_inj_nominal.str')

mad.use('leir')
mad.input('twiss, chrom, table=twchr;')

import xtrack as xt

mad_seq = mad.sequence.leir
line = xt.Line.from_madx_sequence(mad_seq)
line.particle_ref = xt.Particles(gamma0=mad_seq.beam.gamma,
                                 mass0=mad_seq.beam.mass * 1e9,
                                 q0=mad_seq.beam.charge)

line.configure_bend_model(core='full', edge='full')
tw = line.twiss(method='4d')

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

tptc = mad.table.ptc_twiss
tptc_p = mad.table.ptc_twiss_pdp
tptc_m = mad.table.ptc_twiss_mdp

betx = 0.5 * (tptc_p.beta11 + tptc_m.beta11)
alfx = 0.5 * (tptc_p.alfa11 + tptc_m.alfa11)
d_betx = (tptc_p.beta11 - tptc_m.beta11)/ (2 * delta_chrom)
d_alfx = (tptc_p.alfa11 - tptc_m.alfa11)/ (2 * delta_chrom)

bx_ptc = d_betx / betx
ax_ptc = d_alfx - alfx * bx_ptc

bety = 0.5 * (tptc_p.beta22 + tptc_m.beta22)
alfy = 0.5 * (tptc_p.alfa22 + tptc_m.alfa22)
d_bety = (tptc_p.beta22 - tptc_m.beta22)/ (2 * delta_chrom)
d_alfy = (tptc_p.alfa22 - tptc_m.alfa22)/ (2 * delta_chrom)

by_ptc = d_bety / bety
ay_ptc = d_alfy - alfy * by_ptc

import matplotlib.pyplot as plt
plt.close('all')

plt.figure(1, figsize=(6.4, 4.8 * 1.5))

ax1 = plt.subplot(4,1,1)
plt.plot(tptc.s, ax_ptc, label='ptc')
plt.plot(tw.s, tw.ax_chrom, label='xsuite')
plt.ylabel(r'$A_x$')
plt.legend(loc='best')

plt.subplot(4,1,2, sharex=ax1)
plt.plot(tptc.s, bx_ptc)
plt.plot(tw.s, tw.bx_chrom)
plt.ylabel(r'$B_x$')

plt.subplot(4,1,3, sharex=ax1)
plt.plot(tptc.s, ay_ptc)
plt.plot(tw.s, tw.ay_chrom)
plt.ylabel(r'$A_y$')

plt.subplot(4,1,4, sharex=ax1)
plt.plot(tptc.s, by_ptc)
plt.plot(tw.s, tw.by_chrom)
plt.ylabel(r'$B_y$')
plt.xlabel('s [m]')

# Same for beta and orbit
plt.figure(2, figsize=(6.4, 4.8 * 1.5))

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

