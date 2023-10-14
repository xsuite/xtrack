from cpymad.madx import Madx
import xtrack as xt
import xpart as xp
import xdeps as xd

import numpy as np

# hllhc15 can be found at git@github.com:lhcopt/hllhc15.git

mad = Madx()

orbit_settings = {
    'acbh19.r3b1': 15e-6,
    'acbv20.r3b1': 10e-6,
}

mad.input(f"""
call,file="../../test_data/hllhc15_thick/lhc.seq";
call,file="../../test_data/hllhc15_thick/hllhc_sequence.madx";
seqedit,sequence=lhcb1;flatten;cycle,start=IP7;flatten;endedit;
seqedit,sequence=lhcb2;flatten;cycle,start=IP7;flatten;endedit;
beam, sequence=lhcb1, particle=proton, pc=7000;
call,file="../../test_data/hllhc15_thick/opt_round_150_1500.madx";
""")
mad.globals.update(orbit_settings)

mad.use(sequence="lhcb1")
seq = mad.sequence.lhcb1
mad.twiss()

collider = xt.Multiline.from_json(
    '../../test_data/hllhc15_thick/hllhc15_collider_thick.json')
collider.vars.update(orbit_settings)
line = collider.lhcb1
line.twiss_default['method'] = '4d'
line.build_tracker()

ele_start='ip3'
ele_stop='ip4'

xs_map = xt.SecondOrderTaylorMap.from_line(
    line, ele_start=ele_start, ele_stop=ele_stop)
TT = xs_map.T
RR = xs_map.R
k = xs_map.k

mad.input('''
select, flag=sectormap, pattern='ip';
twiss, sectormap, sectorpure, sectortable=secttab;
''')

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

# Check T
for ii in range(6):
    for jj in range(6):
        for kk in range(6):
            scaled_tt = (TT[ii, jj, kk]
                         / scale_out[ii] * scale_in[jj] * scale_in[kk])
            scaled_tt_mad = (sectmad[f't{ii+1}{jj+1}{kk+1}', ele_stop]
                             / scale_out[ii] * scale_in[jj] * scale_in[kk])
            # The following means that a change of one sigma in jj, kk results
            # in an error of less than 5e-4 sigmas on ii
            assert np.isclose(scaled_tt, scaled_tt_mad, atol=5e-4, rtol=0)

# Check R
for ii in range(6):
    for jj in range(6):
        scaled_rr = RR[ii, jj] / scale_out[ii] * scale_in[jj]
        scaled_rr_mad = sectmad[f'r{ii+1}{jj+1}', ele_stop] * (
                            scale_in[jj] / scale_out[ii])
        # The following means that a change of one sigma in jj results
        # in an error of less than 5e-4 sigmas on ii
        assert np.isclose(scaled_rr, scaled_rr_mad, atol=5e-4, rtol=0)

# Check k
for ii in range(6):
    scaled_k = k[ii] / scale_out[ii]
    scaled_k_mad = sectmad[f'k{ii+1}', ele_stop] / scale_out[ii]
    # The following means that a the orbit kick is the same within 5e-5 sigmas
    assert np.isclose(scaled_k, scaled_k_mad, atol=5e-5, rtol=0)