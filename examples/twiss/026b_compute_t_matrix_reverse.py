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
beam, sequence=lhcb1, particle=proton, pc=7000;
beam, sequence=lhcb2, particle=proton, pc=7000, bv=-1;
call,file="../../test_data/hllhc15_thick/opt_round_150_1500.madx";

acbv16.l7b2 = 1e-5;
acbh14.r7b2 = 8e-6;
""")

mad.use(sequence="lhcb2")
seq = mad.sequence.lhcb2
mad.twiss()

mad.input('''

seqedit,sequence=lhcb2;flatten;reflect;flatten;endedit;

use, sequence=lhcb2;
select, flag=sectormap, pattern='ip';
twiss, sectormap, sectorpure, sectortable=secttab_pure;
twiss, sectormap, sectortable=secttab;
''')

sectmad  = xd.Table(mad.table.secttab)
sectpure = xd.Table(mad.table.secttab_pure)

collider = xt.Multiline.from_json('../../test_data/hllhc15_thick/hllhc15_collider_thick.json')

line = collider.lhcb2
line.particle_ref = xp.Particles(mass0=seq.beam.mass*1e9, gamma0=seq.beam.gamma)
line.twiss_default['method'] = '4d'
line.twiss_default['reverse'] = False
line.build_tracker()

line.vv['acbv16.l7b2'] = 1e-5
line.vv['acbh14.r7b2'] = 8e-6

ele_start='ip4'
ele_stop='ip3'

TT = line.compute_T_matrix(ele_start=ele_start, ele_stop=ele_stop)
twinit = line.twiss().get_twiss_init(ele_start)
RR = line.compute_one_turn_matrix_finite_differences(ele_start=ele_start, ele_stop=ele_stop,
                                                     particle_on_co=twinit.particle_on_co,
                                                     )['R_matrix']

RR_reflect = RR.copy()
RR_reflect[0, :] *= -1
RR_reflect[1, :] *= -1
RR_reflect[:, 0] *= -1
RR_reflect[:, 1] *= -1

TT_reflect = TT.copy()
TT_reflect[0, :, :] *= -1
TT_reflect[1, :, :] *= -1
TT_reflect[:, 0, :] *= -1
TT_reflect[:, 1, :] *= -1
TT_reflect[:, :, 0] *= -1
TT_reflect[:, :, 1] *= -1

RR_mad = np.zeros((6,6))
for ii in range(6):
    for jj in range(6):
        RR_mad[ii, jj] = sectmad[f'r{ii+1}{jj+1}', 'ip3']

TT_mad = np.zeros((6,6,6))
for ii in range(6):
    for jj in range(6):
        for kk in range(6):
            TT_mad[ii, jj, kk] = sectmad[f't{ii+1}{jj+1}{kk+1}', 'ip3']


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
            scaled_tt = TT_reflect[ii, jj, kk] / scale_out[ii] * scale_in[jj] * scale_in[kk]
            scaled_tt_mad = TT_mad[ii, jj, kk] / scale_out[ii] * scale_in[jj] * scale_in[kk]

            assert np.isclose(scaled_tt, scaled_tt_mad, atol=0.0005, rtol=0)