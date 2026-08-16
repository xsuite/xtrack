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


def _install_toy_multibunch_beambeam():
    shared_ips = {'ip1': xt.Marker(), 'ip2': xt.Marker()}
    env = xt.Environment(lines={
        'cw': _make_toy_ring('cw', shared_ips),
        'acw': _make_toy_ring('acw', shared_ips),
    })

    filling_scheme_cw = np.zeros(N_SLOTS, dtype=int)
    filling_scheme_acw = np.zeros(N_SLOTS, dtype=int)
    filling_scheme_cw[[0, 2, 5]] = 1
    filling_scheme_acw[[0, 3, 6]] = 1
    intensity_cw = np.zeros(N_SLOTS)
    intensity_acw = np.zeros(N_SLOTS)
    intensity_cw[[0, 2, 5]] = [1.0e11, 2.0e11, 3.0e11]
    intensity_acw[[0, 3, 6]] = [1.5e11, 2.5e11, 3.5e11]

    setup = env.xfields.install_multibunch_beambeam(
        clockwise_line='cw',
        anticlockwise_line='acw',
        ips=['ip1', 'ip2'],
        num_long_range_encounters_per_side=1,
        harmonic_number=N_SLOTS,
        bunch_spacing_buckets=1,
        nemitt_x=NEMITT_X,
        nemitt_y=NEMITT_Y,
        filling_scheme_cw=filling_scheme_cw,
        filling_scheme_acw=filling_scheme_acw,
        bunch_intensity_particles_cw=intensity_cw,
        bunch_intensity_particles_acw=intensity_acw,
        survey_separation=False,
    )
    return (env, setup, filling_scheme_cw, filling_scheme_acw,
            intensity_cw, intensity_acw)


def test_multibunch_beambeam_toy_installation_and_setup():
    # Characterize the current standalone path through a normalized set of
    # encounter, element and solution properties. The same checks can later be
    # applied to the consolidated install/configure path.
    (env, setup, filling_scheme_cw, filling_scheme_acw,
     intensity_cw, intensity_acw) = _install_toy_multibunch_beambeam()
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

    zeta_cw = -setup.filled_slots_cw * SLOT_LENGTH
    zeta_acw = -setup.filled_slots_acw * SLOT_LENGTH
    gamma0 = float(env.cw.particle_ref.gamma0[0])
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

        # Element grids are increasing in zeta for binary search, hence the
        # reverse of public physical-slot order under zeta = -slot * spacing.
        xo.assert_allclose(bb_cw.own_beam_zeta, zeta_cw[::-1], rtol=0, atol=0)
        xo.assert_allclose(bb_acw.own_beam_zeta, zeta_acw[::-1], rtol=0, atol=0)
        assert len(bb_cw.other_beam_zeta) == len(setup.filled_slots_acw)
        assert len(bb_acw.other_beam_zeta) == len(setup.filled_slots_cw)

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
        assert bb.num_other_bunches == len(setup.filled_slots_acw)
        xo.assert_allclose(
            bb.other_beam_num_particles[:bb.num_other_bunches],
            setup.bunch_intensity_particles_acw[::-1], rtol=0, atol=0)
    for bb in reduced.bb_acw.values():
        assert bb.num_other_bunches == len(setup.filled_slots_cw)
        xo.assert_allclose(
            bb.other_beam_num_particles[:bb.num_other_bunches],
            setup.bunch_intensity_particles_cw[::-1], rtol=0, atol=0)

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
            n_bunches = int(reduced_bb.num_other_bunches)
            assert full_bb.num_other_bunches == n_bunches
            for field in (
                    'other_beam_zeta', 'other_beam_x', 'other_beam_y',
                    'other_beam_num_particles'):
                xo.assert_allclose(
                    getattr(full_bb, field)[:n_bunches],
                    getattr(reduced_bb, field)[:n_bunches],
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
    for base in expected_encounters:
        name_cw = reduced.bb_name(base, mirror=False)
        name_acw = reduced.bb_name(base, mirror=True)
        sigma_x_cw = np.sqrt(
            mbtw_cw_dyn['betx', name_cw] * NEMITT_X / gamma0_cw)
        sigma_x_acw = np.sqrt(
            mbtw_acw_dyn['betx', name_acw] * NEMITT_X / gamma0_acw)
        xo.assert_allclose(reduced.bb_cw[base].sigma_x,
                           sigma_x_cw[::-1], rtol=1e-14)
        xo.assert_allclose(reduced.bb_acw[base].sigma_x,
                           sigma_x_acw[::-1], rtol=1e-14)
        xo.assert_allclose(reduced.bb_cw[base].other_beam_sigma_x,
                           sigma_x_acw[::-1], rtol=1e-14)
        xo.assert_allclose(reduced.bb_acw[base].other_beam_sigma_x,
                           sigma_x_cw[::-1], rtol=1e-14)

    # A changed filling count requires differently sized Xobjects. The setup
    # rebuilds the elements under the same line names, preserving geometry and
    # the shared strength knob; every array then has exactly the new own or
    # opposing bunch count.
    filling_scheme_cw_new = np.zeros(N_SLOTS, dtype=int)
    filling_scheme_acw_new = np.zeros(N_SLOTS, dtype=int)
    filling_scheme_cw_new[[1, 4]] = 1
    filling_scheme_acw_new[[0, 2, 5, 7]] = 1
    intensity_cw_new = np.zeros(N_SLOTS)
    intensity_acw_new = np.zeros(N_SLOTS)
    intensity_cw_new[[1, 4]] = [1.2e11, 2.2e11]
    intensity_acw_new[[0, 2, 5, 7]] = [1.4e11, 2.4e11, 3.4e11, 4.4e11]
    env.cw['beambeam_scale'] = 0.29
    setup.set_filling(
        filling_scheme_cw=filling_scheme_cw_new,
        filling_scheme_acw=filling_scheme_acw_new,
        bunch_intensity_particles_cw=intensity_cw_new,
        bunch_intensity_particles_acw=intensity_acw_new)

    xo.assert_allclose(setup.filled_slots_cw, [1, 4], rtol=0, atol=0)
    xo.assert_allclose(setup.filled_slots_acw, [0, 2, 5, 7], rtol=0, atol=0)
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
