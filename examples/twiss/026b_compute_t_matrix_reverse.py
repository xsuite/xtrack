from cpymad.madx import Madx
import xtrack as xt
import xpart as xp
import xdeps as xd

import numpy as np

# hllhc15 can be found at git@github.com:lhcopt/hllhc15.git

kill_fringes_and_edges = True

mad = Madx()

mad.input(f"""
call,file="../../test_data/hllhc15_thick/lhc.seq";
call,file="../../test_data/hllhc15_thick/hllhc_sequence.madx";
seqedit,sequence=lhcb1;flatten;cycle,start=IP7;flatten;endedit;
seqedit,sequence=lhcb2;flatten;cycle,start=IP7;flatten;endedit;
beam, sequence=lhcb1, particle=proton, pc=7000;
beam, sequence=lhcb2, particle=proton, pc=7000, bv=-1;
call,file="../../test_data/hllhc15_thick/opt_round_150_1500.madx";
""")

mad.use(sequence="lhcb2")
seq = mad.sequence.lhcb2
mad.twiss()

mad.input('''
seqedit,sequence=lhcb2;flatten;reflect;flatten;endedit;
use, sequence=lhcb2;
select, flag=sectormap, pattern='ip';
twiss, sectormap, sectorpure, sectortable=secttab;
''')

collider = xt.Multiline.from_json('../../test_data/hllhc15_thick/hllhc15_collider_thick.json')

line = collider.lhcb2
line.particle_ref = xp.Particles(mass0=seq.beam.mass*1e9, gamma0=seq.beam.gamma)
line.twiss_default['method'] = '4d'
line.twiss_default['reverse'] = False
line.build_tracker()

ele_start='ip4'
ele_stop='ip3'

TT = line.compute_T_matrix(ele_start=ele_start, ele_stop=ele_stop)
twinit = line.twiss().get_twiss_init(ele_start)
RR = line.compute_one_turn_matrix_finite_differences(ele_start=ele_start, ele_stop=ele_stop,
                                                     particle_on_co=twinit.particle_on_co,
                                                     )['R_matrix']

sectmad  = xd.Table(mad.table.secttab)
