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
line.config.XTRACK_USE_EXACT_DRIFTS = True
tw0 = line.twiss(method='4d')
tw1 = line.twiss(method='4d', delta0=1e-3)
tw0_sp = line.twiss(method='4d', start=line.element_names[0], end='_end_point',
                   betx=1, bety=1, delta=0)
tw1_sp = line.twiss(method='4d', start=line.element_names[0], end='_end_point',
                     betx=1, bety=1, delta=1e-3)
mad.input(f'''
  ptc_create_universe;
  !ptc_create_layout, time=false, model=2, exact=true, method=6, nst=10;
  ptc_create_layout, time=false, model=1, exact=true, method=6, nst=100;
    select, flag=ptc_twiss, clear;
    select, flag=ptc_twiss, column=name,keyword,s,l,x,px,y,py,beta11,beta22,disp1,k1l;
    ptc_twiss, closed_orbit, icase=56, no=2, deltap=0, table=ptc_twiss,
               summary_table=ptc_twiss_summary, slice_magnets=true;
    ptc_twiss, closed_orbit, icase=56, no=2, deltap=0, table=ttt,
               x={tw0.x[0]}, px={tw0.px[0]}, y={tw0.y[0]}, py={tw0.py[0]},
               pt={tw0.delta[0]}, !!!!! NB pt is delta in PTC
               betx={tw0.betx[0]}, alfx={tw0.alfx[0]}, bety={tw0.bety[0]}, alfy={tw0.alfy[0]},
               betz=1,
               dx={tw0.dx[0]}, dpx={tw0.dpx[0]}, dy={tw0.dy[0]}, dpy={tw0.dpy[0]},
               summary_table=ttt_summ_0, slice_magnets=true;
    ptc_twiss, closed_orbit, icase=56, no=2, deltap=0, table=ttt_p
               x={tw1.x[0]}, px={tw1.px[0]}, y={tw1.y[0]}, py={tw1.py[0]},
               pt={tw1.delta[0]}, !!!!! NB pt is delta in PTC
               betx={tw1.betx[0]}, alfx={tw1.alfx[0]}, bety={tw1.bety[0]}, alfy={tw1.alfy[0]},
               betz=1,
               dx={tw1.dx[0]}, dpx={tw1.dpx[0]}, dy={tw1.dy[0]}, dpy={tw1.dpy[0]},
               summary_table=ttt_summ_p, slice_magnets=true;
    ptc_twiss, closed_orbit, icase=56, no=2, deltap=0, table=ttt_sp
                x={tw0_sp.x[0]}, px={tw0_sp.px[0]}, y={tw0_sp.y[0]}, py={tw0_sp.py[0]},
                pt={tw0_sp.delta[0]}, !!!!! NB pt is delta in PTC
                betx={tw0_sp.betx[0]}, alfx={tw0_sp.alfx[0]}, bety={tw0_sp.bety[0]}, alfy={tw0_sp.alfy[0]},
                betz=1,
                dx={tw0_sp.dx[0]}, dpx={tw0_sp.dpx[0]}, dy={tw0_sp.dy[0]}, dpy={tw0_sp.dpy[0]},
                summary_table=ttt_summ_sp, slice_magnets=true;
    ptc_twiss, closed_orbit, icase=56, no=2, deltap=0, table=ttt_sp_p
                x={tw1_sp.x[0]}, px={tw1_sp.px[0]}, y={tw1_sp.y[0]}, py={tw1_sp.py[0]},
                pt={tw1_sp.delta[0]}, !!!!! NB pt is delta in PTC
                betx={tw1_sp.betx[0]}, alfx={tw1_sp.alfx[0]}, bety={tw1_sp.bety[0]}, alfy={tw1_sp.alfy[0]},
                betz=1,
                dx={tw1_sp.dx[0]}, dpx={tw1_sp.dpx[0]}, dy={tw1_sp.dy[0]}, dpy={tw1_sp.dpy[0]},
                summary_table=ttt_summ_sp_p, slice_magnets=true;
  ptc_end;
''')

tp0 = mad.table.ttt
tp1 = mad.table.ttt_p
tp0_sp = mad.table.ttt_sp
tp1_sp = mad.table.ttt_sp_p

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.plot(tp1.s, tp1.x, label='ptc')
plt.plot(tw1.s, tw1.x, label='xsuite')

plt.figure(2)
plt.plot(tp1.s, tp1.betx/tp0.betx-1, label='ptc')
plt.plot(tw1.s, tw1.betx/tw0.betx-1, label='xsuite')

plt.figure(3)
plt.plot(tp1_sp.s, tp1_sp.x, label='ptc')
plt.plot(tw1_sp.s, tw1_sp.x, label='xsuite')

plt.figure(4)
plt.plot(tp1_sp.s, tp1_sp.betx/tp0_sp.betx-1, label='ptc')
plt.plot(tw1_sp.s, tw1_sp.betx/tw0_sp.betx-1, label='xsuite')

plt.show()