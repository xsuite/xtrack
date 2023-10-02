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

line = xt.Line.from_madx_sequence(mad.sequence.lhcb1,
            allow_thick=True, deferred_expressions=True)
line.particle_ref = xp.Particles(mass0=seq.beam.mass*1e9, gamma0=seq.beam.gamma)
line.twiss_default['method'] = '4d'

tw = line.twiss()

ele_start = 'ip3'
ele_stop = 'ip4'

twinit = tw.get_twiss_init(ele_start)

R_plus = {}
R_minus = {}

steps0 = {'x': 1e-6, 'px': 1e-7, 'y': 1e-6, 'py': 1e-7, 'zeta': 1e-7, 'ptau': 1e-7}
factor_step = 10

steps = {kk: factor_step * vv for kk, vv in steps0.items()}

for kk in ['x', 'px', 'y', 'py', 'zeta', 'ptau']:

    ti_plus = twinit.copy()
    setattr(ti_plus.particle_on_co, kk,
        getattr(twinit.particle_on_co, kk) + steps[kk])

    tw_plus = line.twiss(ele_start=ele_start, ele_stop=ele_stop,
                         twiss_init=ti_plus)
    R_plus[kk] = tw_plus.get_R_matrix(ele_start=ele_start, ele_stop=ele_stop)

    ti_minus = twinit.copy()
    setattr(ti_minus.particle_on_co, kk,
        getattr(twinit.particle_on_co, kk) - steps[kk])

    tw_minus = line.twiss(ele_start=ele_start, ele_stop=ele_stop,
                            twiss_init=ti_minus)
    R_minus[kk] = tw_minus.get_R_matrix(ele_start=ele_start, ele_stop=ele_stop)

TT = np.zeros((6, 6, 6))

TT[:, :, 0] = 0.5 * (R_plus['x'] - R_minus['x']) / (2 * steps['x'])
TT[:, :, 1] = 0.5 * (R_plus['px'] - R_minus['px']) / (2 * steps['px'])
TT[:, :, 2] = 0.5 * (R_plus['y'] - R_minus['y']) / (2 * steps['y'])
TT[:, :, 3] = 0.5 * (R_plus['py'] - R_minus['py']) / (2 * steps['py'])
TT[:, :, 4] = 0.5 * (R_plus['zeta'] - R_minus['zeta']) / (2 * steps['zeta'])
TT[:, :, 5] = 0.5 * (R_plus['ptau'] - R_minus['ptau']) / (2 * steps['ptau'])

sectmad  = xd.Table(mad.table.secttab)

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
        if jj == 4: continue
        for kk in range(6):
            if kk == 4: continue
            scaled_tt = TT[ii, jj, kk] / scale_out[ii] * scale_in[jj] * scale_in[kk]
            scaled_tt_mad = sectmad[f't{ii+1}{jj+1}{kk+1}', ele_stop] / scale_out[ii] * scale_in[jj] * scale_in[kk]

            assert np.isclose(scaled_tt, scaled_tt_mad, atol=0.03, rtol=0)