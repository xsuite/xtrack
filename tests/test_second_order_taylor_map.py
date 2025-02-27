# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import pathlib

import numpy as np
from cpymad.madx import Madx

import xobjects as xo
import xtrack as xt
from xobjects.test_helpers import for_all_test_contexts

test_data_folder = pathlib.Path(
        __file__).parent.joinpath('../test_data').absolute()

@for_all_test_contexts
def test_line_with_second_order_maps(test_context):

    line = xt.Line.from_json(test_data_folder /
                             'hllhc15_thick/lhc_thick_with_knobs.json')
    line.build_tracker(_context=test_context)
    line.vars['vrf400'] = 16
    line.vars['lagrf400.b1'] = 0.5

    line.vars['acbh22.l7b1'] = 15e-6
    line.vars['acbv21.l7b1'] = 10e-6

    ele_cut = ['ip1', 'ip2', 'ip3', 'ip4', 'ip5', 'ip6', 'ip7']

    line_maps = line.get_line_with_second_order_maps(split_at=ele_cut)
    line_maps.build_tracker(_context=test_context)

    line_maps.get_table().show()

    tw = line.twiss()
    tw_map = line_maps.twiss()

    xo.assert_allclose(tw_map.rows[ele_cut].s, tw.rows[ele_cut].s, rtol=0, atol=1e-12)

    xo.assert_allclose(tw_map.rows[ele_cut].x, tw.rows[ele_cut].x, rtol=0, atol=1e-12)
    xo.assert_allclose(tw_map.rows[ele_cut].px, tw.rows[ele_cut].px, rtol=0, atol=1e-12)
    xo.assert_allclose(tw_map.rows[ele_cut].y, tw.rows[ele_cut].y, rtol=0, atol=1e-12)
    xo.assert_allclose(tw_map.rows[ele_cut].py, tw.rows[ele_cut].py, rtol=0, atol=1e-12)
    xo.assert_allclose(tw_map.rows[ele_cut].zeta, tw.rows[ele_cut].zeta, rtol=0, atol=1e-10)
    xo.assert_allclose(tw_map.rows[ele_cut].delta, tw.rows[ele_cut].delta, rtol=0, atol=1e-12)

    xo.assert_allclose(tw_map.rows[ele_cut].betx, tw.rows[ele_cut].betx, rtol=1e-5, atol=0)
    xo.assert_allclose(tw_map.rows[ele_cut].alfx, tw.rows[ele_cut].alfx, rtol=1e-5, atol=1e-6)
    xo.assert_allclose(tw_map.rows[ele_cut].bety, tw.rows[ele_cut].bety, rtol=1e-5, atol=0)
    xo.assert_allclose(tw_map.rows[ele_cut].alfy, tw.rows[ele_cut].alfy, rtol=1e-5, atol=1e-6)

    xo.assert_allclose(np.mod(tw_map.qx, 1), np.mod(tw.qx, 1), rtol=0, atol=1e-7)
    xo.assert_allclose(np.mod(tw_map.qy, 1), np.mod(tw.qy, 1), rtol=0, atol=1e-7)
    xo.assert_allclose(tw_map.dqx, tw.dqx, rtol=0, atol=5e-2)
    xo.assert_allclose(tw_map.dqy, tw.dqy, rtol=0, atol=5e-2)
    xo.assert_allclose(tw_map.c_minus, tw.c_minus, rtol=0, atol=1e-5)
    xo.assert_allclose(tw_map.circumference, tw.circumference, rtol=0, atol=5e-9)


@for_all_test_contexts
def test_second_order_maps_against_madx(test_context):


    orbit_settings = {
        'acbh19.r3b1': 15e-6,
        'acbv20.r3b1': 10e-6,
        'acbv19.r3b2': 15e-6,
        'acbh20.r3b2': 10e-6,
    }

    # Generate Xsuite maps

    collider = xt.Environment.from_json(test_data_folder /
                            'hllhc15_thick/hllhc15_collider_thick.json')
    collider.vars.update(orbit_settings)
    collider['lhcb1'].twiss_default['method'] = '4d'
    collider['lhcb2'].twiss_default['method'] = '4d'
    collider.build_trackers(_context=test_context)

    map_b1 = xt.SecondOrderTaylorMap.from_line(
        line=collider.lhcb1, start='ip3', end='ip4')

    map_b4 = xt.SecondOrderTaylorMap.from_line(
        line=collider.lhcb2, start='ip4', end='ip3')
    map_b2_reflected = map_b4.scale_coordinates(scale_x=-1, scale_px=-1)

    # Generate MAD-X maps

    mad = Madx(stdout=False)
    mad.call(str(test_data_folder / "hllhc15_thick/lhc.seq"))
    mad.call(str(test_data_folder / "hllhc15_thick/hllhc_sequence.madx"))

    mad.input("""
        beam, sequence=lhcb1, particle=proton, pc=7000;
        beam, sequence=lhcb2, particle=proton, pc=7000, bv=-1;
    """)

    mad.call(str(test_data_folder / "hllhc15_thick/opt_round_150_1500.madx"))
    mad.globals.update(orbit_settings)

    mad.use(sequence="lhcb1")
    seq = mad.sequence.lhcb1
    mad.twiss()

    mad.input('''
    select, flag=sectormap, pattern='ip';
    twiss, sectormap, sectorpure, sectortable=secttab_b1;
    ''')
    sectmad_b1  = xt.Table(mad.table.secttab_b1)

    mad.input('''
        seqedit,sequence=lhcb2;flatten;reflect;flatten;endedit;
        use, sequence=lhcb2;
    ''')

    mad.input('''
    select, flag=sectormap, pattern='ip';
    twiss, sectormap, sectorpure, sectortable=secttab_b2;
    ''')
    sectmad_b2  = xt.Table(mad.table.secttab_b2)

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
            xo.assert_allclose(scaled_k, scaled_k_mad, atol=5e-5, rtol=0)

        # Check R
        for ii in range(6):
            for jj in range(6):
                scaled_rr = RR[ii, jj] / scale_out[ii] * scale_in[jj]
                scaled_rr_mad = sectmad[f'r{ii+1}{jj+1}', end] * (
                                    scale_in[jj] / scale_out[ii])
                # The following means that a change of one sigma in jj results
                # in an error of less than 5e-4 sigmas on ii
                xo.assert_allclose(scaled_rr, scaled_rr_mad, atol=5e-4, rtol=0)

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
                    xo.assert_allclose(scaled_tt, scaled_tt_mad, atol=5e-4, rtol=0)


