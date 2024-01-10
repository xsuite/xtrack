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

tw0 = line.twiss(method='4d')
line.configure_bend_model(core='full', edge='full')
tw = line.twiss(method='4d')

mad.input('''
  ptc_create_universe;
  !ptc_create_layout, time=false, model=2, exact=true, method=6, nst=10;
  ptc_create_layout, time=false, model=1, exact=true, method=6, nst=10;
    select, flag=ptc_twiss, clear;
    select, flag=ptc_twiss, column=name,keyword,s,l,x,px,y,py,beta11,beta22,disp1,k1l;
    ptc_twiss, closed_orbit, icase=56, no=2, deltap=0, table=ptc_twiss,
               summary_table=ptc_twiss_summary, slice_magnets=true;
    ptc_twiss, closed_orbit, icase=56, no=2, deltap=1e-3, table=ptc_twiss_dp,
               summary_table=ptc_twiss_summary_dp, slice_magnets=true;
  ptc_end;
''')

tptc = mad.table.ptc_twiss

