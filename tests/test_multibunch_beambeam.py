# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

import numpy as np
import pytest

import xfields as xf
import xobjects as xo
import xtrack as xt
from xtrack.multibunch_beambeam import RigidBunchBBSetup


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


def _make_toy_environment():
    shared_ips = {'ip1': xt.Marker(), 'ip2': xt.Marker()}
    return xt.Environment(lines={
        'cw': _make_toy_ring('cw', shared_ips),
        'acw': _make_toy_ring('acw', shared_ips),
    })


def _toy_filling():
    filling_scheme_cw = np.zeros(N_SLOTS, dtype=int)
    filling_scheme_acw = np.zeros(N_SLOTS, dtype=int)
    filling_scheme_cw[[0, 2, 5]] = 1
    filling_scheme_acw[[0, 3, 6]] = 1
    intensity_cw = np.zeros(N_SLOTS)
    intensity_acw = np.zeros(N_SLOTS)
    intensity_cw[[0, 2, 5]] = [1.0e11, 2.0e11, 3.0e11]
    intensity_acw[[0, 3, 6]] = [1.5e11, 2.5e11, 3.5e11]
    return (filling_scheme_cw, filling_scheme_acw,
            intensity_cw, intensity_acw)


def _install_toy_rigid_bunch_beambeam():
    env = _make_toy_environment()
    (filling_scheme_cw, filling_scheme_acw,
     intensity_cw, intensity_acw) = _toy_filling()

    env.xfields.install_beambeam_interactions(
        clockwise_line='cw', anticlockwise_line='acw',
        ip_names=['ip1', 'ip2'],
        num_long_range_encounters_per_side=1,
        harmonic_number=N_SLOTS,
        bunch_spacing_buckets=1,
        mode='rigid_bunch',
        survey_separation=True)
    assert env._bb_config['mode'] == 'rigid_bunch'
    bb_names_cw = [name for name in env.cw.element_names
                   if name.startswith('bb_')]
    bb_names_acw = [name for name in env.acw.element_names
                    if name.startswith('bb_')]
    assert len(bb_names_cw) == 6
    assert len(bb_names_acw) == 6
    for name in bb_names_cw:
        assert len(env.cw[name].own_beam_zeta) == N_SLOTS
        assert len(env.cw[name].other_beam_zeta) == N_SLOTS
    for name in bb_names_acw:
        assert len(env.acw[name].own_beam_zeta) == N_SLOTS
        assert len(env.acw[name].other_beam_zeta) == N_SLOTS

    # Installation and its full-slot elements are serializable before filling-
    # dependent physics parameters are loaded by configuration.
    env = xt.Environment.from_dict(env.to_dict())
    setup = env.xfields.configure_beambeam_interactions(
        nemitt_x=NEMITT_X,
        nemitt_y=NEMITT_Y,
        filling_scheme_cw=filling_scheme_cw,
        filling_scheme_acw=filling_scheme_acw,
        bunch_intensity_particles_cw=intensity_cw,
        bunch_intensity_particles_acw=intensity_acw)
    return (env, setup, filling_scheme_cw, filling_scheme_acw,
            intensity_cw, intensity_acw)


def test_multibunch_beambeam_toy_installation_and_setup():
    # Characterize the consolidated install/configure path through a normalized
    # set of encounter, element and solution properties.
    (env, setup, filling_scheme_cw, filling_scheme_acw,
     intensity_cw, intensity_acw) = _install_toy_rigid_bunch_beambeam()
    assert isinstance(setup, RigidBunchBBSetup)

    expected_encounters = [
        'bb_ip1_ho', 'bb_ip1_r01', 'bb_ip1_l01',
        'bb_ip2_ho', 'bb_ip2_r01', 'bb_ip2_l01',
    ]
    assert setup.enc_names == expected_encounters
    assert setup.ip_offsets == {'ip1': 0, 'ip2': 6}
    assert setup.n_slots == N_SLOTS
    assert setup.bunch_spacing_zeta == SLOT_LENGTH
    xo.assert_allclose(setup.filling_scheme_cw, filling_scheme_cw,
                       rtol=0, atol=0)
    xo.assert_allclose(setup.filling_scheme_acw, filling_scheme_acw,
                       rtol=0, atol=0)
    xo.assert_allclose(setup.filled_slots_cw, [0, 2, 5], rtol=0, atol=0)
    xo.assert_allclose(setup.filled_slots_acw, [0, 3, 6], rtol=0, atol=0)
    xo.assert_allclose(
        setup.bunch_intensity_particles_cw,
        intensity_cw[[0, 2, 5]], rtol=0, atol=0)
    xo.assert_allclose(
        setup.bunch_intensity_particles_acw,
        intensity_acw[[0, 3, 6]], rtol=0, atol=0)

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

    beta0_cw = float(env.cw.particle_ref.beta0[0])
    gamma0_cw = float(env.cw.particle_ref.gamma0[0])
    beta0_acw = float(env.acw.particle_ref.beta0[0])
    gamma0_acw = float(env.acw.particle_ref.gamma0[0])
    for base, offset in zip(expected_encounters, expected_offsets):
        bb_cw = setup.bb_cw[base]
        bb_acw = setup.bb_acw[base]
        assert isinstance(bb_cw, xf.BeamBeamBiGaussianRigidBunch2D)
        assert isinstance(bb_acw, xf.BeamBeamBiGaussianRigidBunch2D)
        assert bb_cw.coherent == 1
        assert bb_acw.coherent == 1
        assert bb_cw.zeta_period == N_SLOTS * SLOT_LENGTH
        assert bb_acw.zeta_period == N_SLOTS * SLOT_LENGTH
        assert bb_cw.zeta_match_tol == 0.1 * SLOT_LENGTH
        assert bb_acw.zeta_match_tol == 0.1 * SLOT_LENGTH
        assert bb_cw.zeta_offset == -offset * SLOT_LENGTH
        assert bb_acw.zeta_offset == offset * SLOT_LENGTH

        # Element grids cover every RF slot and increase in zeta for binary
        # search, hence they reverse the public physical-slot order.
        all_zeta = -np.arange(N_SLOTS) * SLOT_LENGTH
        xo.assert_allclose(bb_cw.own_beam_zeta, all_zeta[::-1], rtol=0, atol=0)
        xo.assert_allclose(bb_acw.own_beam_zeta, all_zeta[::-1], rtol=0, atol=0)
        assert len(bb_cw.other_beam_zeta) == N_SLOTS
        assert len(bb_acw.other_beam_zeta) == N_SLOTS

        geom = setup.geom[base]
        # The shared covariance API includes relativistic beta in the
        # normalized-to-geometric emittance conversion.
        xo.assert_allclose(
            geom['sigma_x_cw'], np.sqrt(
                geom['betx_cw'] * NEMITT_X / (beta0_cw * gamma0_cw)),
            rtol=1e-14)
        xo.assert_allclose(
            geom['sigma_y_cw'], np.sqrt(
                geom['bety_cw'] * NEMITT_Y / (beta0_cw * gamma0_cw)),
            rtol=1e-14)
        xo.assert_allclose(
            geom['sigma_x_acw'], np.sqrt(
                geom['betx_acw'] * NEMITT_X / (beta0_acw * gamma0_acw)),
            rtol=1e-14)
        xo.assert_allclose(
            geom['sigma_y_acw'], np.sqrt(
                geom['bety_acw'] * NEMITT_Y / (beta0_acw * gamma0_acw)),
            rtol=1e-14)
        xo.assert_allclose(
            bb_cw.sigma_x, geom['sigma_x_cw'], rtol=0, atol=0)
        xo.assert_allclose(
            bb_cw.sigma_y, geom['sigma_y_cw'], rtol=0, atol=0)
        xo.assert_allclose(
            bb_cw.other_beam_sigma_x, geom['sigma_x_acw'], rtol=0, atol=0)
        xo.assert_allclose(
            bb_cw.other_beam_sigma_y, geom['sigma_y_acw'], rtol=0, atol=0)

    # The setup geometry is the normalized view of the shared Xfields result.
    shared_geometry, _ = xf.compute_beambeam_geometry(
        encounter_table=setup.encounter_table,
        line_cw=env.cw, line_acw=env.acw,
        element_names_cw=setup.bb_names_cw,
        element_names_acw=setup.bb_names_acw,
        nemitt_x=NEMITT_X, nemitt_y=NEMITT_Y,
        survey_separation=False)
    for ii, base in enumerate(expected_encounters):
        geom = setup.geom[base]
        row = shared_geometry.iloc[ii]
        for field in ('betx_cw', 'bety_cw', 'betx_acw', 'bety_acw'):
            xo.assert_allclose(geom[field], row[field], rtol=0, atol=0)
        xo.assert_allclose(
            geom['sigma_x_cw'], np.sqrt(row['Sigma_11_cw']), rtol=0, atol=0)
        xo.assert_allclose(
            geom['sigma_y_cw'], np.sqrt(row['Sigma_33_cw']), rtol=0, atol=0)
        xo.assert_allclose(
            geom['sigma_x_acw'], np.sqrt(row['Sigma_11_acw']), rtol=0, atol=0)
        xo.assert_allclose(
            geom['sigma_y_acw'], np.sqrt(row['Sigma_33_acw']), rtol=0, atol=0)
        xo.assert_allclose(geom['sep_x'], row['separation_x'], rtol=0, atol=0)
        xo.assert_allclose(geom['sep_y'], row['separation_y'], rtol=0, atol=0)

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
            reduced.cw_line[name], xf.BeamBeamBiGaussianRigidBunch2D)
    for name in reduced.bb_names_acw:
        assert isinstance(
            reduced.acw_line[name], xf.BeamBeamBiGaussianRigidBunch2D)

    mbtw_cw, mbtw_acw = reduced.solve(
        max_iterations=2,
        tol_sigma=0,
        twiss_mode='fast_orbit',
        show_progress=False,
    )
    assert len(mbtw_cw) == len(setup.filled_slots_cw)
    assert len(mbtw_acw) == len(setup.filled_slots_acw)

    for bb in reduced.bb_cw.values():
        assert bb.num_other_bunches == N_SLOTS
        xo.assert_allclose(
            bb.other_beam_num_particles,
            intensity_acw[::-1], rtol=0, atol=0)
    for bb in reduced.bb_acw.values():
        assert bb.num_other_bunches == N_SLOTS
        xo.assert_allclose(
            bb.other_beam_num_particles,
            intensity_cw[::-1], rtol=0, atol=0)

    # At the +1-slot encounter, physical slot 0 faces an empty slot and gets no
    # kick, while the remaining two bunches have partners. This checks the
    # offset-sign conversion together with the public negative-zeta convention.
    for mirror, bb in (
            (False, reduced.bb_cw['bb_ip1_r01']),
            (True, reduced.bb_acw['bb_ip1_r01'])):
        probe = xt.Particles(
            p0c=7e12,
            x=np.full(3, 1.0e-3),
            zeta=reduced.bunch_zeta(mirror))
        bb.track(probe)
        assert probe.px[0] == 0
        assert np.all(np.abs(probe.px[1:]) > 0)

    setup.load_solution(mbtw_cw, mbtw_acw)
    for base in expected_encounters:
        for full_bb, reduced_bb in (
                (setup.bb_cw[base], reduced.bb_cw[base]),
                (setup.bb_acw[base], reduced.bb_acw[base])):
            assert full_bb.num_other_bunches == N_SLOTS
            for field in (
                    'other_beam_zeta', 'other_beam_x', 'other_beam_y',
                    'other_beam_num_particles'):
                xo.assert_allclose(
                    getattr(full_bb, field),
                    getattr(reduced_bb, field),
                    rtol=0, atol=0)

    # Dynamic-beta sizes arrive in public filled-slot order from Twiss and are
    # reordered together with the negative-zeta grids inside each element.
    mbtw_cw_dyn, mbtw_acw_dyn = reduced.solve(
        max_iterations=1,
        tol_sigma=0,
        dynamic_beta=True,
        twiss_mode='fast',
        show_progress=False,
    )
    gamma0_cw = float(reduced.cw_line.particle_ref.gamma0[0])
    gamma0_acw = float(reduced.acw_line.particle_ref.gamma0[0])
    indices_cw = N_SLOTS - 1 - setup.filled_slots_cw
    indices_acw = N_SLOTS - 1 - setup.filled_slots_acw
    for base in expected_encounters:
        name_cw = reduced.bb_name(base, mirror=False)
        name_acw = reduced.bb_name(base, mirror=True)
        sigma_x_cw = np.sqrt(
            mbtw_cw_dyn['betx', name_cw] * NEMITT_X / gamma0_cw)
        sigma_x_acw = np.sqrt(
            mbtw_acw_dyn['betx', name_acw] * NEMITT_X / gamma0_acw)
        xo.assert_allclose(np.asarray(reduced.bb_cw[base].sigma_x)[indices_cw],
                           sigma_x_cw, rtol=1e-14)
        xo.assert_allclose(
            np.asarray(reduced.bb_acw[base].sigma_x)[indices_acw],
                           sigma_x_acw, rtol=1e-14)
        xo.assert_allclose(
            np.asarray(reduced.bb_cw[base].other_beam_sigma_x)[indices_acw],
            sigma_x_acw, rtol=1e-14)
        xo.assert_allclose(
            np.asarray(reduced.bb_acw[base].other_beam_sigma_x)[indices_cw],
            sigma_x_cw, rtol=1e-14)

    # A changed filling updates the full-slot arrays in place. No element is
    # rebuilt, and empty slots retain zero opposing population.
    filling_scheme_cw_new = np.zeros(N_SLOTS, dtype=int)
    filling_scheme_acw_new = np.zeros(N_SLOTS, dtype=int)
    filling_scheme_cw_new[[1, 4]] = 1
    filling_scheme_acw_new[[0, 2, 5, 7]] = 1
    intensity_cw_new = np.zeros(N_SLOTS)
    intensity_acw_new = np.zeros(N_SLOTS)
    intensity_cw_new[[1, 4]] = [1.2e11, 2.2e11]
    intensity_acw_new[[0, 2, 5, 7]] = [1.4e11, 2.4e11, 3.4e11, 4.4e11]
    env.cw['beambeam_scale'] = 0.29
    original_cw = dict(setup.bb_cw)
    original_acw = dict(setup.bb_acw)
    setup.set_filling(
        filling_scheme_cw=filling_scheme_cw_new,
        filling_scheme_acw=filling_scheme_acw_new,
        bunch_intensity_particles_cw=intensity_cw_new,
        bunch_intensity_particles_acw=intensity_acw_new)

    xo.assert_allclose(setup.filled_slots_cw, [1, 4], rtol=0, atol=0)
    xo.assert_allclose(setup.filled_slots_acw, [0, 2, 5, 7], rtol=0, atol=0)
    for base, bb in setup.bb_cw.items():
        assert bb is original_cw[base]
        assert len(bb.own_beam_zeta) == N_SLOTS
        assert len(bb.other_beam_zeta) == N_SLOTS
        assert bb.num_own_bunches == N_SLOTS
        assert bb.num_other_bunches == N_SLOTS
        assert bb.scale_strength == 0.29
        xo.assert_allclose(
            bb.other_beam_num_particles, np.zeros(N_SLOTS), rtol=0, atol=0)
    for base, bb in setup.bb_acw.items():
        assert bb is original_acw[base]
        assert len(bb.own_beam_zeta) == N_SLOTS
        assert len(bb.other_beam_zeta) == N_SLOTS
        assert bb.num_own_bunches == N_SLOTS
        assert bb.num_other_bunches == N_SLOTS
        assert bb.scale_strength == 0.29
        xo.assert_allclose(
            bb.other_beam_num_particles, np.zeros(N_SLOTS), rtol=0, atol=0)


def test_rigid_bunch_pattern_contract_matches_beam_stats_monitor():
    # Beam-beam and BeamStatsMonitor interpret filling schemes as occupancy
    # patterns over physical slots. Slot i is centred at
    # zeta = -i * bunch_spacing_zeta; intensities are a separate input.
    filling_scheme = np.array([1, 0, 1, 1])
    bunch_spacing_zeta = 5.0
    slot_intensities = np.array([1.0e11, 0.0, 2.0e11, 3.0e11])

    line = xt.Line(
        elements=[xt.Drift(length=4 * bunch_spacing_zeta)],
        particle_ref=xt.Particles(p0c=7e12))
    setup = RigidBunchBBSetup(
        line, line, ips=[], num_long_range_encounters_per_side=0,
        harmonic_number=4, bunch_spacing_buckets=1,
        nemitt_x=NEMITT_X, nemitt_y=NEMITT_Y)
    setup.set_filling(
        filling_scheme_cw=filling_scheme,
        filling_scheme_acw=filling_scheme,
        bunch_intensity_particles_cw=4.0e11,
        bunch_intensity_particles_acw=slot_intensities)

    monitor = xt.BeamStatsMonitor(
        filling_scheme=filling_scheme,
        selected_slots=[0, 3],
        bunch_spacing_zeta=bunch_spacing_zeta)

    xo.assert_allclose(setup.filled_slots_cw, [0, 2, 3], rtol=0, atol=0)
    xo.assert_allclose(setup.filled_slots_cw, monitor.filled_slots,
                       rtol=0, atol=0)
    xo.assert_allclose(setup.bunch_zeta(mirror=False), [0, -10, -15],
                       rtol=0, atol=0)
    xo.assert_allclose(
        monitor.zeta_centers_unwrapped(line_length=20)[0], [0, -15],
        rtol=0, atol=0)
    xo.assert_allclose(setup.bunch_intensity_particles_cw,
                       np.full(3, 4.0e11), rtol=0, atol=0)
    xo.assert_allclose(setup.bunch_intensity_particles_acw,
                       [1.0e11, 2.0e11, 3.0e11], rtol=0, atol=0)

    with pytest.raises(ValueError, match='slot-indexed array'):
        setup.set_filling(
            filling_scheme_cw=filling_scheme,
            filling_scheme_acw=filling_scheme,
            bunch_intensity_particles_cw=[1.0e11, 2.0e11, 3.0e11],
            bunch_intensity_particles_acw=1.0e11)
