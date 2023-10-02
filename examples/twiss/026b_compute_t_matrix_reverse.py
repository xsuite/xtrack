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
call,file="../../test_data/hllhc15_thick/opt_round_150_1500.madx";
""")

mad.use(sequence="lhcb1")
seq = mad.sequence.lhcb1
mad.twiss()

mad.input('''
select, flag=sectormap, pattern='ip';
twiss, sectormap, sectorpure, sectortable=secttab;
''')

collider = xt.Multiline.from_json('../../test_data/hllhc15_thick/hllhc15_collider_thick.json')

line = collider.lhcb1
line.particle_ref = xp.Particles(mass0=seq.beam.mass*1e9, gamma0=seq.beam.gamma)
line.twiss_default['method'] = '4d'
line.build_tracker()

ele_start='ip3'
ele_stop='ip4'

TT = line.compute_T_matrix(ele_start=ele_start, ele_stop=ele_stop)

sectmad  = xd.Table(mad.table.secttab)

tw = line.twiss()

nemitt_x = 2.5e-6
nemitt_y = 2.5e-6
scale_in = [
    np.sqrt(tw['betx', ele_start] * nemitt_x / tw['gamma0']),
    np.sqrt(tw['gamx', ele_start] * nemitt_x / tw['gamma0']),
    np.sqrt(tw['bety', ele_start] * nemitt_y / tw['gamma0']),
    np.sqrt(tw['gamy', ele_start] * nemitt_y / tw['gamma0']),
    0.05,
    1e-3]

scale_out = [
    np.sqrt(tw['betx', ele_stop] * nemitt_x / tw['gamma0']),
    np.sqrt(tw['gamx', ele_stop] * nemitt_x / tw['gamma0']),
    np.sqrt(tw['bety', ele_stop] * nemitt_y / tw['gamma0']),
    np.sqrt(tw['gamy', ele_stop] * nemitt_y / tw['gamma0']),
    0.05,
    1e-3]

for ii in range(6):
    for jj in range(6):
        for kk in range(6):
            scaled_tt = TT[ii, jj, kk] / scale_out[ii] * scale_in[jj] * scale_in[kk]
            scaled_tt_mad = sectmad[f't{ii+1}{jj+1}{kk+1}', ele_stop] / scale_out[ii] * scale_in[jj] * scale_in[kk]

            assert np.isclose(scaled_tt, scaled_tt_mad, atol=0.0005, rtol=0)