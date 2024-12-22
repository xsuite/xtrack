from cpymad.madx import Madx
import xtrack as xt
import xdeps as xd

import numpy as np

orbit_settings = {
    'acbh19.r3b1': 15e-6,
    'acbv20.r3b1': 10e-6,
    'acbv19.r3b2': 15e-6,
    'acbh20.r3b2': 10e-6,
}

# Generate Xsuite maps

collider = xt.Environment.from_json(
    '../../test_data/hllhc15_thick/hllhc15_collider_thick.json')
collider.vars.update(orbit_settings)
collider['lhcb1'].twiss_default['method'] = '4d'
collider['lhcb2'].twiss_default['method'] = '4d'

map_b1 = xt.SecondOrderTaylorMap.from_line(
    line=collider.lhcb1, start='ip3', end='ip4')

map_b4 = xt.SecondOrderTaylorMap.from_line(
    line=collider.lhcb2, start='ip4', end='ip3')
map_b2_reflected = map_b4.scale_coordinates(scale_x=-1, scale_px=-1)

# Generate MAD-X maps

mad = Madx()
mad.input(f"""
call,file="../../test_data/hllhc15_thick/lhc.seq";
call,file="../../test_data/hllhc15_thick/hllhc_sequence.madx";
beam, sequence=lhcb1, particle=proton, pc=7000;
beam, sequence=lhcb2, particle=proton, pc=7000, bv=-1;
call,file="../../test_data/hllhc15_thick/opt_round_150_1500.madx";
""")
mad.globals.update(orbit_settings)

mad.use(sequence="lhcb1")
seq = mad.sequence.lhcb1
mad.twiss()

mad.input('''
select, flag=sectormap, pattern='ip';
twiss, sectormap, sectorpure, sectortable=secttab_b1;
''')
sectmad_b1  = xd.Table(mad.table.secttab_b1)

mad.input('''
    seqedit,sequence=lhcb2;flatten;reflect;flatten;endedit;
    use, sequence=lhcb2;
''')

mad.input('''
select, flag=sectormap, pattern='ip';
twiss, sectormap, sectorpure, sectortable=secttab_b2;
''')
sectmad_b2  = xd.Table(mad.table.secttab_b2)

# Compare

for line_name in ['lhcb1', 'lhcb2']:

    if line_name == 'lhcb1':
        xs_map = map_b1
        sectmad = sectmad_b1
        start = 'ip3'
        end = 'ip4'
        tw = collider.lhcb1.twiss()
    elif line_name == 'lhcb2':
        xs_map = map_b2_reflected
        sectmad = sectmad_b2
        start = 'ip4'
        end = 'ip3'
        tw = collider.lhcb2.twiss()
    else:
        raise ValueError(f'Unknown line_name: {line_name}')

    TT = xs_map.T
    RR = xs_map.R
    k = xs_map.k


    nemitt_x = 2.5e-6
    nemitt_y = 2.5e-6
    scale_in = [
        np.sqrt(tw['betx', start] * nemitt_x / tw['gamma0']),
        np.sqrt(tw['gamx', start] * nemitt_x / tw['gamma0']),
        np.sqrt(tw['bety', start] * nemitt_y / tw['gamma0']),
        np.sqrt(tw['gamy', start] * nemitt_y / tw['gamma0']),
        0.05,
        1e-3]

    scale_out = [
        np.sqrt(tw['betx', end] * nemitt_x / tw['gamma0']),
        np.sqrt(tw['gamx', end] * nemitt_x / tw['gamma0']),
        np.sqrt(tw['bety', end] * nemitt_y / tw['gamma0']),
        np.sqrt(tw['gamy', end] * nemitt_y / tw['gamma0']),
        0.05,
        1e-3]

    # Check k
    for ii in range(6):
        scaled_k = k[ii] / scale_out[ii]
        scaled_k_mad = sectmad[f'k{ii+1}', end] / scale_out[ii]
        # The following means that a the orbit kick is the same within 5e-5 sigmas
        assert np.isclose(scaled_k, scaled_k_mad, atol=5e-5, rtol=0)

    # Check R
    for ii in range(6):
        for jj in range(6):
            scaled_rr = RR[ii, jj] / scale_out[ii] * scale_in[jj]
            scaled_rr_mad = sectmad[f'r{ii+1}{jj+1}', end] * (
                                scale_in[jj] / scale_out[ii])
            # The following means that a change of one sigma in jj results
            # in an error of less than 5e-4 sigmas on ii
            assert np.isclose(scaled_rr, scaled_rr_mad, atol=5e-4, rtol=0)

    # Check T
    for ii in range(6):
        for jj in range(6):
            for kk in range(6):
                scaled_tt = (TT[ii, jj, kk]
                            / scale_out[ii] * scale_in[jj] * scale_in[kk])
                scaled_tt_mad = (sectmad[f't{ii+1}{jj+1}{kk+1}', end]
                                / scale_out[ii] * scale_in[jj] * scale_in[kk])
                # The following means that a change of one sigma in jj, kk results
                # in an error of less than 5e-4 sigmas on ii
                assert np.isclose(scaled_tt, scaled_tt_mad, atol=5e-4, rtol=0)



