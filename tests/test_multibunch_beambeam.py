# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

import numpy as np

import xfields as xf
import xobjects as xo
import xtrack as xt


N_CELLS = 8
CELL_LENGTH = 10.0
N_SLOTS = 8
SLOT_LENGTH = CELL_LENGTH * N_CELLS / N_SLOTS
NEMITT_X = 2.0e-6
NEMITT_Y = 2.5e-6


def _make_toy_ring(suffix, shared_ips):
    """Stable FODO ring with boundaries at every possible BB encounter."""
    elements = []
    names = []
    for ii in range(N_CELLS):
        if ii == 0:
            marker_name = 'ip1'
        elif ii == 3:
            marker_name = 'ip2'
        else:
            marker_name = f'cell{ii}_{suffix}'
        marker = shared_ips.get(marker_name, xt.Marker())

        # The installer places encounters 1 um downstream of their nominal
        # position. These small drifts and markers make those positions element
        # boundaries, including the +/- half-slot long-range positions.
        elements.extend([
            marker,
            xt.Multipole(knl=[0, 0.05]),
            xt.Drift(length=1e-6),
            xt.Marker(),
            xt.Drift(length=0.5 * CELL_LENGTH - 1e-6),
            xt.Multipole(knl=[0, -0.05]),
            xt.Drift(length=1e-6),
            xt.Marker(),
            xt.Drift(length=0.5 * CELL_LENGTH - 1e-6),
        ])
        names.extend([
            marker_name,
            f'qf{ii}_{suffix}',
            f'deps_a{ii}_{suffix}',
            f'edge_a{ii}_{suffix}',
            f'drift_a{ii}_{suffix}',
            f'qd{ii}_{suffix}',
            f'deps_b{ii}_{suffix}',
            f'edge_b{ii}_{suffix}',
            f'drift_b{ii}_{suffix}',
        ])

    line = xt.Line(
        elements=elements,
        element_names=names,
        particle_ref=xt.Particles(p0c=7e12),
    )
    line.twiss_default['method'] = '4d'
    return line


def _install_toy_multibunch_beambeam():
    shared_ips = {'ip1': xt.Marker(), 'ip2': xt.Marker()}
    env = xt.Environment(lines={
        'cw': _make_toy_ring('cw', shared_ips),
        'acw': _make_toy_ring('acw', shared_ips),
    })

    filling_cw = np.zeros(N_SLOTS)
    filling_acw = np.zeros(N_SLOTS)
    filling_cw[[0, 2, 5]] = [1.0e11, 2.0e11, 3.0e11]
    filling_acw[[0, 3, 6]] = [1.5e11, 2.5e11, 3.5e11]

    setup = env.xfields.install_multibunch_beambeam(
        clockwise_line='cw',
        anticlockwise_line='acw',
        ips=['ip1', 'ip2'],
        num_long_range_encounters_per_side=1,
        harmonic_number=N_SLOTS,
        bunch_spacing_buckets=1,
        nemitt_x=NEMITT_X,
        nemitt_y=NEMITT_Y,
        filling_clockwise=filling_cw,
        filling_anticlockwise=filling_acw,
        survey_separation=False,
    )
    return env, setup, filling_cw, filling_acw


def test_multibunch_beambeam_toy_installation_and_setup():
    # Characterize the current standalone path through a normalized set of
    # encounter, element and solution properties. The same checks can later be
    # applied to the consolidated install/configure path.
    env, setup, filling_cw, filling_acw = _install_toy_multibunch_beambeam()

    expected_encounters = [
        'bb_ip1_ho', 'bb_ip1_r01', 'bb_ip1_l01',
        'bb_ip2_ho', 'bb_ip2_r01', 'bb_ip2_l01',
    ]
    assert setup.enc_names == expected_encounters
    assert setup.ip_offsets == {'ip1': 0, 'ip2': 6}
    assert setup.n_slots == N_SLOTS
    assert setup.slot_len == SLOT_LENGTH
    xo.assert_allclose(setup.bunches_cw, [0, 2, 5], rtol=0, atol=0)
    xo.assert_allclose(setup.bunches_acw, [0, 3, 6], rtol=0, atol=0)
    xo.assert_allclose(
        setup.num_particles_cw, filling_cw[[0, 2, 5]], rtol=0, atol=0)
    xo.assert_allclose(
        setup.num_particles_acw, filling_acw[[0, 3, 6]], rtol=0, atol=0)

    expected_offsets = [0, 1, 7, 6, 7, 5]
    assert [setup.geom[name]['offset'] for name in expected_encounters] \
        == expected_offsets
    assert [setup.geom[name]['signed_n'] for name in expected_encounters] \
        == [0, 1, -1, 0, 1, -1]
    for geom in setup.geom.values():
        assert geom['sep_x'] == 0
        assert geom['sep_y'] == 0

    xo.assert_allclose(
        env.cw.get_table()['s', setup.bb_names_cw],
        [1e-6, 5.000001, 75.000001, 30.000001, 35.000001, 25.000001],
        rtol=0, atol=2e-14)
    xo.assert_allclose(
        env.acw.get_table()['s', setup.bb_names_acw],
        [1e-6, 75.000001, 5.000001, 30.000001, 25.000001, 35.000001],
        rtol=0, atol=2e-14)

    zeta_cw = setup.bunches_cw * SLOT_LENGTH
    zeta_acw = setup.bunches_acw * SLOT_LENGTH
    gamma0 = float(env.cw.particle_ref.gamma0[0])
    for base, offset in zip(expected_encounters, expected_offsets):
        bb_cw = setup.bb_cw[base]
        bb_acw = setup.bb_acw[base]
        assert isinstance(bb_cw, xf.BeamBeamBiGaussianMultibunch2D)
        assert isinstance(bb_acw, xf.BeamBeamBiGaussianMultibunch2D)
        assert bb_cw.coherent == 1
        assert bb_acw.coherent == 1
        assert bb_cw.zeta_period == N_SLOTS * SLOT_LENGTH
        assert bb_acw.zeta_period == N_SLOTS * SLOT_LENGTH
        assert bb_cw.zeta_match_tol == 0.1 * SLOT_LENGTH
        assert bb_acw.zeta_match_tol == 0.1 * SLOT_LENGTH
        assert bb_cw.zeta_offset == offset * SLOT_LENGTH
        assert bb_acw.zeta_offset == -offset * SLOT_LENGTH

        xo.assert_allclose(bb_cw.own_beam_zeta, zeta_cw, rtol=0, atol=0)
        xo.assert_allclose(bb_acw.own_beam_zeta, zeta_acw, rtol=0, atol=0)
        assert len(bb_cw.other_beam_zeta) == len(setup.bunches_acw)
        assert len(bb_acw.other_beam_zeta) == len(setup.bunches_cw)

        geom = setup.geom[base]
        xo.assert_allclose(
            bb_cw.sigma_x,
            np.sqrt(geom['betx_cw'] * NEMITT_X / gamma0), rtol=1e-14)
        xo.assert_allclose(
            bb_cw.sigma_y,
            np.sqrt(geom['bety_cw'] * NEMITT_Y / gamma0), rtol=1e-14)
        xo.assert_allclose(
            bb_cw.other_beam_sigma_x,
            np.sqrt(geom['betx_acw'] * NEMITT_X / gamma0), rtol=1e-14)
        xo.assert_allclose(
            bb_cw.other_beam_sigma_y,
            np.sqrt(geom['bety_acw'] * NEMITT_Y / gamma0), rtol=1e-14)

    env.cw['beambeam_scale'] = 0.37
    for name in setup.bb_names_cw:
        assert env.cw[name].scale_strength == 0.37
    for name in setup.bb_names_acw:
        assert env.acw[name].scale_strength == 0.37
    env.cw['beambeam_scale'] = 1.0

    reduced = setup.second_order_maps()
    assert reduced.enc_names == setup.enc_names
    assert reduced.geom == setup.geom
    for name in reduced.bb_names_cw:
        assert isinstance(
            reduced.cw_line[name], xf.BeamBeamBiGaussianMultibunch2D)
    for name in reduced.bb_names_acw:
        assert isinstance(
            reduced.acw_line[name], xf.BeamBeamBiGaussianMultibunch2D)

    mbtw_cw, mbtw_acw = reduced.solve(
        max_iterations=2,
        tol_sigma=0,
        twiss_mode='fast_orbit',
        show_progress=False,
    )
    assert len(mbtw_cw) == len(setup.bunches_cw)
    assert len(mbtw_acw) == len(setup.bunches_acw)

    for bb in reduced.bb_cw.values():
        assert bb.num_other_bunches == len(setup.bunches_acw)
        xo.assert_allclose(
            bb.other_beam_num_particles[:bb.num_other_bunches],
            setup.num_particles_acw, rtol=0, atol=0)
    for bb in reduced.bb_acw.values():
        assert bb.num_other_bunches == len(setup.bunches_cw)
        xo.assert_allclose(
            bb.other_beam_num_particles[:bb.num_other_bunches],
            setup.num_particles_cw, rtol=0, atol=0)

    setup.load_solution(mbtw_cw, mbtw_acw)
    for base in expected_encounters:
        for full_bb, reduced_bb in (
                (setup.bb_cw[base], reduced.bb_cw[base]),
                (setup.bb_acw[base], reduced.bb_acw[base])):
            n_bunches = int(reduced_bb.num_other_bunches)
            assert full_bb.num_other_bunches == n_bunches
            for field in (
                    'other_beam_zeta', 'other_beam_x', 'other_beam_y',
                    'other_beam_num_particles'):
                xo.assert_allclose(
                    getattr(full_bb, field)[:n_bunches],
                    getattr(reduced_bb, field)[:n_bunches],
                    rtol=0, atol=0)

    # A changed filling count requires differently sized Xobjects. The setup
    # rebuilds the elements under the same line names, preserving geometry and
    # the shared strength knob; every array then has exactly the new own or
    # opposing bunch count.
    filling_cw_new = np.zeros(N_SLOTS)
    filling_acw_new = np.zeros(N_SLOTS)
    filling_cw_new[[1, 4]] = [1.2e11, 2.2e11]
    filling_acw_new[[0, 2, 5, 7]] = [1.4e11, 2.4e11, 3.4e11, 4.4e11]
    env.cw['beambeam_scale'] = 0.29
    setup.set_filling(filling_cw_new, filling_acw_new)

    xo.assert_allclose(setup.bunches_cw, [1, 4], rtol=0, atol=0)
    xo.assert_allclose(setup.bunches_acw, [0, 2, 5, 7], rtol=0, atol=0)
    for bb in setup.bb_cw.values():
        assert len(bb.own_beam_zeta) == 2
        assert len(bb.other_beam_zeta) == 4
        assert bb.num_own_bunches == 2
        assert bb.num_other_bunches == 4
        assert bb.scale_strength == 0.29
        xo.assert_allclose(
            bb.other_beam_num_particles, np.zeros(4), rtol=0, atol=0)
    for bb in setup.bb_acw.values():
        assert len(bb.own_beam_zeta) == 4
        assert len(bb.other_beam_zeta) == 2
        assert bb.num_own_bunches == 4
        assert bb.num_other_bunches == 2
        assert bb.scale_strength == 0.29
        xo.assert_allclose(
            bb.other_beam_num_particles, np.zeros(2), rtol=0, atol=0)
