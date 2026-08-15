# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import pathlib

import numpy as np
import pytest
from cpymad.madx import Madx

import xobjects as xo
import xtrack as xt
from xobjects.test_helpers import for_all_test_contexts

test_data_folder = pathlib.Path(
        __file__).parent.joinpath('../test_data').absolute()

@for_all_test_contexts
def test_line_with_second_order_maps(test_context):

    line = xt.load(test_data_folder /
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
    xo.assert_allclose(tw_map.line_length, tw.line_length, rtol=0, atol=5e-9)


@for_all_test_contexts
def test_line_with_second_order_maps_split_at_octupoles(test_context):

    line = xt.load(test_data_folder /
                             'hllhc15_thick/lhc_thick_with_knobs.json')
    line.build_tracker(_context=test_context)
    line.vars['vrf400'] = 16
    line.vars['lagrf400.b1'] = 0.5

    # power the Landau octupoles (strong amplitude detuning)
    for vv in line.vars.keys():
        if vv.startswith('kof.') or vv.startswith('kod.'):
            line.vars[vv] = 40.

    mo_names = [nn for nn in line.element_names if nn.startswith('mo.')]
    assert len(mo_names) == 168

    ele_cut = ['ip1', 'ip2', 'ip3', 'ip4', 'ip5', 'ip6', 'ip7']

    # splitting also at the octupoles keeps them as exact elements between
    # the maps (split elements are excluded from the maps)
    line_maps_keep = line.get_line_with_second_order_maps(
        split_at=ele_cut + mo_names)
    line_maps = line.get_line_with_second_order_maps(split_at=ele_cut)
    for lm in (line_maps_keep, line_maps):
        lm.build_tracker(_context=test_context)

    # the kept octupoles are present as exact elements
    assert isinstance(line_maps_keep[mo_names[0]], xt.Octupole)
    assert mo_names[0] not in line_maps.element_names

    tw = line.twiss()
    tw_map = line_maps_keep.twiss()

    xo.assert_allclose(tw_map.rows[ele_cut].s, tw.rows[ele_cut].s, rtol=0, atol=1e-12)
    xo.assert_allclose(tw_map.rows[ele_cut].x, tw.rows[ele_cut].x, rtol=0, atol=1e-12)
    xo.assert_allclose(tw_map.rows[ele_cut].px, tw.rows[ele_cut].px, rtol=0, atol=1e-12)
    xo.assert_allclose(tw_map.rows[ele_cut].y, tw.rows[ele_cut].y, rtol=0, atol=1e-12)
    xo.assert_allclose(tw_map.rows[ele_cut].py, tw.rows[ele_cut].py, rtol=0, atol=1e-12)
    xo.assert_allclose(tw_map.rows[ele_cut].betx, tw.rows[ele_cut].betx, rtol=1e-5, atol=0)
    xo.assert_allclose(tw_map.rows[ele_cut].bety, tw.rows[ele_cut].bety, rtol=1e-5, atol=0)
    xo.assert_allclose(np.mod(tw_map.qx, 1), np.mod(tw.qx, 1), rtol=0, atol=1e-7)
    xo.assert_allclose(np.mod(tw_map.qy, 1), np.mod(tw.qy, 1), rtol=0, atol=1e-7)
    xo.assert_allclose(tw_map.dqx, tw.dqx, rtol=0, atol=5e-2)
    xo.assert_allclose(tw_map.dqy, tw.dqy, rtol=0, atol=5e-2)

    # amplitude detuning is preserved by the kept octupoles (the small
    # residual difference is the sextupole-driven detuning, which second
    # order maps cannot carry), while it is absent from the plain map line:
    # launch test particles at small and high amplitude and measure their
    # tunes from the turn-by-turn data
    def _tune_fft(xs):
        """Fractional tune from turn-by-turn data (Hann window + parabolic
        interpolation of the FFT peak)."""
        xs = xs - np.mean(xs)
        ff = np.abs(np.fft.rfft(xs * np.hanning(len(xs))))
        ii = np.argmax(ff[1:-1]) + 1
        dd = 0.5 * (ff[ii - 1] - ff[ii + 1]) / (ff[ii - 1] - 2 * ff[ii] + ff[ii + 1])
        return (ii + dd) / len(xs)

    dq = {}
    for label, ll in (('ref', line), ('keep', line_maps_keep),
                      ('maps', line_maps)):
        pp = ll.build_particles(x_norm=[0.5, 5., 0.5], y_norm=[0.5, 0.5, 5.],
                                nemitt_x=2.5e-6, nemitt_y=2.5e-6)
        ll.track(pp, num_turns=512, turn_by_turn_monitor=True,
                 freeze_longitudinal=True)
        mon = ll.record_last_track
        ctx2np = test_context.nparray_from_context_array
        qx = [_tune_fft(ctx2np(mon.x)[jj, :]) for jj in range(3)]
        qy = [_tune_fft(ctx2np(mon.y)[jj, :]) for jj in range(3)]
        dq[label] = (qx[1] - qx[0], qy[2] - qy[0])  # amplitude detuning

    assert dq['ref'][0] > 5e-3   # measured: ~1e-2
    assert dq['ref'][1] > 5e-3
    xo.assert_allclose(dq['keep'], dq['ref'], rtol=0, atol=1e-3)
    xo.assert_allclose(dq['maps'], (0, 0), rtol=0, atol=5e-4)


@for_all_test_contexts
def test_second_order_maps_split_at_thick_elements(test_context):

    # FODO ring made only of exactly-linear elements (quadrupoles and
    # expanded drifts) plus two thick octupoles kept exact in the map line
    # by splitting at them: tracking through the map line must reproduce
    # the full line exactly.
    env = xt.Environment()
    env.particle_ref = xt.Particles(p0c=10e9)
    components = []
    for cc in range(8):
        components += [
            env.new(f'qf{cc}', xt.Quadrupole, k1=0.12, length=0.5),
            env.new(f'd1{cc}', xt.Drift, length=2.),
            env.new(f'qd{cc}', xt.Quadrupole, k1=-0.12, length=0.5),
            env.new(f'd2{cc}', xt.Drift, length=2.),
        ]
    # two octupoles back to back (empty span between kept elements) and a
    # marker directly downstream (empty span between keep and split_at)
    components[8:8] = [env.new('mo1', xt.Octupole, k3=3000., length=0.3),
                       env.new('mo2', xt.Octupole, k3=-2000., length=0.3),
                       env.new('m1', xt.Marker)]
    line = env.new_line(components=components)
    line.twiss_default['method'] = '4d'
    line.build_tracker(_context=test_context)

    line_maps_keep = line.get_line_with_second_order_maps(
        split_at=['m1', 'mo1', 'mo2'])
    line_maps = line.get_line_with_second_order_maps(split_at=['m1'])

    # no maps are generated for the empty spans between adjacent cuts
    names = list(line_maps_keep.element_names)
    assert names[names.index('mo1') + 1] == 'mo2'
    assert names[names.index('mo2') + 1] == 'm1'

    for lm in (line_maps_keep, line_maps):
        lm.twiss_default['method'] = '4d'
        lm.build_tracker(_context=test_context)

    tw = line.twiss()
    tw_keep = line_maps_keep.twiss()
    xo.assert_allclose(tw_keep.qx, tw.qx, rtol=0, atol=1e-8)
    xo.assert_allclose(tw_keep.qy, tw.qy, rtol=0, atol=1e-8)

    p_test = dict(x=2e-3, px=1e-5, y=-1.5e-3, py=2e-5)
    res = {}
    for label, ll in (('keep', line_maps_keep), ('maps', line_maps),
                      ('ref', line)):
        pp = ll.build_particles(**p_test)
        ll.track(pp, num_turns=50)
        res[label] = np.array([getattr(pp, cc)[0] for cc in ['x', 'px', 'y', 'py']])

    # kept octupoles -> exact (everything else is linear)
    xo.assert_allclose(res['keep'], res['ref'], rtol=0, atol=1e-12)
    # octupoles inside the maps -> their nonlinearity is lost
    assert np.max(np.abs(res['maps'] - res['ref'])) > 1e-5

    # the original line is not affected by the map lines (shared elements
    # must not be moved out of its buffer)
    pp = line.build_particles(**p_test)
    line.track(pp, num_turns=1)

    with pytest.raises(ValueError, match='not present in the line'):
        line.get_line_with_second_order_maps(split_at=['does_not_exist'])


@for_all_test_contexts
def test_second_order_maps_split_at_repeated_names(test_context):

    # ring with REPEATED element names (same elements placed several times,
    # as it happens e.g. for the drift pieces generated by element
    # insertions): element handling refers to the disambiguated names
    # 'name::N' used in the line/twiss tables. As above, all elements are
    # exactly linear except one thick octupole, so the map line with the
    # octupole split at must reproduce the full line exactly.
    elements = {
        'qf': xt.Quadrupole(k1=0.12, length=0.5),
        'qd': xt.Quadrupole(k1=-0.12, length=0.5),
        'dd': xt.Drift(length=2.),
        'mo': xt.Octupole(k3=3000., length=0.3),
        'm1': xt.Marker(),
    }
    element_names = []
    for cc in range(8):
        element_names += ['qf', 'dd', 'qd', 'dd']
    # octupole surrounded by repeated-name drifts (the map following it
    # starts at a repeated element), marker in the second cell
    element_names[2:2] = ['mo']
    element_names[8:8] = ['m1']
    line = xt.Line(elements=elements, element_names=element_names)
    line.particle_ref = xt.Particles(p0c=10e9)
    line.twiss_default['method'] = '4d'
    line.build_tracker(_context=test_context)

    # split at the octupole (unique name) and at one particular occurrence
    # of the repeated drift (disambiguated name)
    line_maps_keep = line.get_line_with_second_order_maps(
        split_at=['m1', 'mo', 'dd::5'])
    line_maps = line.get_line_with_second_order_maps(split_at=['m1'])

    # the split elements appear in the new line under the same names,
    # including the repeated first element of the line
    for nn in ('qf::0', 'mo', 'm1', 'dd::5'):
        assert nn in line_maps_keep.element_names
    assert isinstance(line_maps_keep['dd::5'], xt.Drift)

    for lm in (line_maps_keep, line_maps):
        lm.twiss_default['method'] = '4d'
        lm.build_tracker(_context=test_context)

    tw = line.twiss()
    tw_keep = line_maps_keep.twiss()
    xo.assert_allclose(tw_keep.qx, tw.qx, rtol=0, atol=1e-8)
    xo.assert_allclose(tw_keep.qy, tw.qy, rtol=0, atol=1e-8)

    p_test = dict(x=2e-3, px=1e-5, y=-1.5e-3, py=2e-5)
    res = {}
    for label, ll in (('keep', line_maps_keep), ('maps', line_maps),
                      ('ref', line)):
        pp = ll.build_particles(**p_test)
        ll.track(pp, num_turns=50)
        res[label] = np.array([getattr(pp, cc)[0] for cc in ['x', 'px', 'y', 'py']])

    # split octupole -> exact (everything else is linear)
    xo.assert_allclose(res['keep'], res['ref'], rtol=0, atol=1e-12)
    # octupole inside the maps -> its nonlinearity is lost
    assert np.max(np.abs(res['maps'] - res['ref'])) > 1e-5

    # a repeated plain name is ambiguous and not accepted
    with pytest.raises(ValueError, match='not present in the line'):
        line.get_line_with_second_order_maps(split_at=['dd'])


@for_all_test_contexts
def test_second_order_maps_split_at_sliced_line(test_context):

    # sliced lattice (thin slices and slice drifts, with parent/replica
    # bookkeeping): the map line built on it must reproduce the sliced line
    # exactly when the only nonlinear element (a thick octupole, excluded
    # from the slicing) is split at.
    env = xt.Environment()
    env.particle_ref = xt.Particles(p0c=10e9)
    components = []
    for cc in range(8):
        components += [
            env.new(f'qf{cc}', xt.Quadrupole, k1=0.12, length=0.5),
            env.new(f'd1{cc}', xt.Drift, length=2.),
            env.new(f'qd{cc}', xt.Quadrupole, k1=-0.12, length=0.5),
            env.new(f'd2{cc}', xt.Drift, length=2.),
        ]
    components[8:8] = [env.new('mo1', xt.Octupole, k3=3000., length=0.3),
                       env.new('m1', xt.Marker)]
    line = env.new_line(components=components)
    line.twiss_default['method'] = '4d'
    line.slice_thick_elements(slicing_strategies=[
        xt.Strategy(slicing=None),  # default: don't slice
        xt.Strategy(slicing=xt.Teapot(4), element_type=xt.Quadrupole),
    ])
    line.build_tracker(_context=test_context)

    line_maps_keep = line.get_line_with_second_order_maps(
        split_at=['m1', 'mo1'])
    line_maps = line.get_line_with_second_order_maps(split_at=['m1'])
    for lm in (line_maps_keep, line_maps):
        lm.twiss_default['method'] = '4d'
        lm.build_tracker(_context=test_context)

    tw = line.twiss()
    tw_keep = line_maps_keep.twiss()
    xo.assert_allclose(tw_keep.qx, tw.qx, rtol=0, atol=1e-8)
    xo.assert_allclose(tw_keep.qy, tw.qy, rtol=0, atol=1e-8)

    p_test = dict(x=2e-3, px=1e-5, y=-1.5e-3, py=2e-5)
    res = {}
    for label, ll in (('keep', line_maps_keep), ('maps', line_maps),
                      ('ref', line)):
        pp = ll.build_particles(**p_test)
        ll.track(pp, num_turns=50)
        res[label] = np.array([getattr(pp, cc)[0] for cc in ['x', 'px', 'y', 'py']])

    # split octupole -> exact (thin quad slices and drifts are linear)
    xo.assert_allclose(res['keep'], res['ref'], rtol=0, atol=1e-12)
    # octupole inside the maps -> its nonlinearity is lost
    assert np.max(np.abs(res['maps'] - res['ref'])) > 1e-5


@for_all_test_contexts
def test_second_order_maps_against_madx(test_context, sandbox_cwd):


    orbit_settings = {
        'acbh19.r3b1': 15e-6,
        'acbv20.r3b1': 10e-6,
        'acbv19.r3b2': 15e-6,
        'acbh20.r3b2': 10e-6,
    }

    # Generate Xsuite maps

    collider = xt.load(test_data_folder /
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
