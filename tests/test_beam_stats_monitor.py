import numpy as np
import pytest
import xobjects as xo
import xpart as xp
from numpy.testing import assert_allclose, assert_equal

from xobjects.test_helpers import (
    allow_no_prebuilt_kernels, for_all_test_contexts)

import xtrack as xt


def _to_numpy(test_context, array):
    return test_context.nparray_from_context_array(array)


@for_all_test_contexts
@allow_no_prebuilt_kernels
def test_beam_stats_monitor_whole_beam_stats(test_context):
    particles = xt.Particles(
        _context=test_context,
        p0c=7e12,
        x=[1., 3., 10.],
        px=[0.1, 0.3, 1.0],
        y=[2., 4., 20.],
        py=[0.2, 0.4, 2.0],
        zeta=[-0.5, -0.25, 0.5],
        delta=[0.01, 0.03, 0.1],
        weight=[2., 1., 4.],
    )
    monitor = xt.BeamStatsMonitor(
        _context=test_context,
        start_at_turn=0,
        stop_at_turn=3,
        every_n_turns=2,
        stats=['num_particles', 'mean_x', 'sigma_x', 'cov_x_px'],
    )

    for _ in range(3):
        monitor.track(particles)
        particles.at_turn += 1

    assert monitor.available_levels == ('beam',)
    assert monitor.default_level == 'beam'
    assert monitor.zeta_centers is None
    assert_equal(monitor.turns, [0, 2])
    assert monitor.mean_x.shape == (2,)
    assert_allclose(monitor.num_particles, [7., 7.])
    assert_allclose(monitor.mean_x, [45. / 7., 45. / 7.])
    assert_allclose(monitor.sigma_x, [np.sqrt(852. / 49.),
                                      np.sqrt(852. / 49.)])
    assert_allclose(monitor.get('mean_x', turn=2), 45. / 7.)


@for_all_test_contexts
@allow_no_prebuilt_kernels
def test_beam_stats_monitor_bunch_stats(test_context):
    particles = xt.Particles(
        _context=test_context,
        p0c=7e12,
        x=[1., 2., 10., 20.],
        px=[0., 0., 0., 0.],
        y=[0., 0., 0., 0.],
        py=[0., 0., 0., 0.],
        zeta=[-0.2, 0.2, -10.2, -9.8],
        delta=[0., 0., 0., 0.],
        weight=[1., 1., 3., 1.],
    )
    monitor = xt.BeamStatsMonitor(
        _context=test_context,
        start_at_turn=0,
        stop_at_turn=1,
        filling_scheme=[1, 1],
        bunch_spacing_zeta=10.,
        stats=['num_particles', 'mean_x'],
    )

    monitor.track(particles)

    assert monitor.available_levels == ('beam', 'bunch')
    assert monitor.default_level == 'bunch'
    assert monitor.mean_x.shape == (1, 2)
    assert_allclose(monitor.num_particles, [[2., 4.]])
    assert_allclose(monitor.mean_x, [[1.5, 12.5]])
    assert_allclose(monitor.get('mean_x', slot=1), [12.5])
    assert_allclose(monitor.get('mean_x', slot=[1, 0]), [[12.5, 1.5]])
    assert_allclose(monitor.get('mean_x', level='beam'), [53. / 6.])
    assert_allclose(monitor.get('mean_x', level='beam', turn=0), 53. / 6.)


@for_all_test_contexts
@allow_no_prebuilt_kernels
def test_beam_stats_monitor_weighted_stats_and_turn_selection(test_context):
    particles = xt.Particles(
        _context=test_context,
        p0c=7e12,
        x=[1., 3., 10.],
        px=[0.1, 0.3, 1.0],
        y=[2., 4., 20.],
        py=[0.2, 0.4, 2.0],
        zeta=[-0.5, -0.25, 0.5],
        delta=[0.01, 0.03, 0.1],
        weight=[2., 1., 4.],
    )
    monitor = xt.BeamStatsMonitor(
        _context=test_context,
        start_at_turn=0,
        stop_at_turn=3,
        every_n_turns=2,
        zeta_range=(-1., 1.),
        num_slices=2,
        stats=[
            'num_particles', 'mean_x', 'sigma_x', 'cov_x_px',
            'gemitt_x_projected', 'nemitt_x_projected'],
    )

    for _ in range(3):
        monitor.track(particles)
        particles.at_turn += 1

    assert_equal(monitor.turns, [0, 2])
    assert monitor.available_levels == ('beam', 'bunch', 'slice')
    assert monitor.default_level == 'slice'
    assert_allclose(monitor.num_particles[:, 0, :], [[3., 4.], [3., 4.]])
    assert_allclose(
        monitor.mean_x[:, 0, :],
        [[5. / 3., 10.], [5. / 3., 10.]])
    assert_allclose(
        monitor.sigma_x[:, 0, :],
        [[np.sqrt(8. / 9.), 0.], [np.sqrt(8. / 9.), 0.]])
    assert_allclose(
        monitor.cov_x_px[:, 0, :],
        [[0.08888888888888888, 0.], [0.08888888888888888, 0.]])
    assert monitor.record_index(2) == 1
    assert monitor.slot_index(0) == 0
    assert monitor.slice_index(0.5) == 1
    assert_allclose(monitor.get('mean_x', turn=2), [[5. / 3., 10.]])
    assert_allclose(monitor.get('mean_x', slot=0), monitor.mean_x[:, 0, :])
    assert_allclose(monitor.get('mean_x', turn=2, slot=0),
                    [5. / 3., 10.])
    assert_allclose(monitor.get('mean_x', turn=2, slot=0, slice_index=1),
                    10.)
    assert monitor.get('mean_x', turn=2, slot=0, slice_index=1,
                       keepdims=True).shape == (1, 1, 1)
    assert_allclose(monitor.get('mean_x', level='bunch'),
                    [[45. / 7.], [45. / 7.]])
    assert_allclose(monitor.get('sigma_x', level='bunch')[:, 0],
                    [np.sqrt(852. / 49.), np.sqrt(852. / 49.)])
    assert_allclose(monitor.get('mean_x', level='beam'),
                    [45. / 7., 45. / 7.])


@for_all_test_contexts
@allow_no_prebuilt_kernels
def test_beam_stats_monitor_pzeta_stats_and_projected_emittance(test_context):
    particles = xt.Particles(
        _context=test_context,
        p0c=1e9,
        x=[0., 0., 0.],
        px=[0., 0., 0.],
        y=[0., 0., 0.],
        py=[0., 0., 0.],
        zeta=[-0.5, 0.25, 1.0],
        pzeta=[0.01, -0.02, 0.05],
        weight=[2., 1., 4.],
    )
    monitor = xt.BeamStatsMonitor(
        _context=test_context,
        start_at_turn=0,
        stop_at_turn=1,
        stats=[
            'num_particles',
            'mean_delta', 'mean_pzeta',
            'sigma_pzeta',
            'cov_zeta_pzeta',
            'gemitt_zeta_projected',
            'nemitt_zeta_projected',
        ],
    )

    monitor.track(particles)

    weights = _to_numpy(test_context, particles.weight)
    zeta = _to_numpy(test_context, particles.zeta)
    delta = _to_numpy(test_context, particles.delta)
    pzeta = _to_numpy(test_context, particles.pzeta)
    mean_zeta = np.average(zeta, weights=weights)
    mean_pzeta = np.average(pzeta, weights=weights)
    var_zeta = np.average(zeta * zeta, weights=weights) - mean_zeta**2
    var_pzeta = np.average(pzeta * pzeta, weights=weights) - mean_pzeta**2
    cov_zeta_pzeta = (
        np.average(zeta * pzeta, weights=weights) - mean_zeta * mean_pzeta)
    gemitt_zeta = np.sqrt(var_zeta * var_pzeta - cov_zeta_pzeta**2)
    beta0_gamma0 = np.average(
        _to_numpy(test_context, particles.beta0 * particles.gamma0),
        weights=weights)

    assert_allclose(monitor.mean_delta, [np.average(delta, weights=weights)])
    assert_allclose(monitor.mean_pzeta, [mean_pzeta])
    assert_allclose(monitor.sigma_pzeta, [np.sqrt(var_pzeta)])
    assert_allclose(monitor.cov_zeta_pzeta, [cov_zeta_pzeta])
    assert_allclose(monitor.gemitt_zeta_projected, [gemitt_zeta])
    assert_allclose(monitor.nemitt_zeta_projected,
                    [gemitt_zeta * beta0_gamma0])


@for_all_test_contexts
def test_beam_stats_monitor_coupled_emittance_and_covariance_optics(
        test_context):
    def w_2d(beta, alpha):
        return np.array([
            [np.sqrt(beta), 0.],
            [-alpha / np.sqrt(beta), 1. / np.sqrt(beta)],
        ])

    gemitt_x = 1.0e-6
    gemitt_y = 2.0e-6
    gemitt_zeta = 3.0e-6
    beta0_gamma0 = 7.0
    weight = 11.0

    betx = 3.0
    alfx = 0.7
    bety = 5.0
    alfy = -0.4
    betzeta = 2.5
    alfzeta = 0.2

    w_matrix = np.eye(6)
    w_matrix[0:2, 0:2] = w_2d(betx, alfx)
    w_matrix[2:4, 2:4] = w_2d(bety, alfy)
    w_matrix[4:6, 4:6] = w_2d(betzeta, alfzeta)
    sigma = w_matrix @ np.diag([
        gemitt_x, gemitt_x,
        gemitt_y, gemitt_y,
        gemitt_zeta, gemitt_zeta,
    ]) @ w_matrix.T

    monitor = xt.BeamStatsMonitor(
        _context=test_context,
        start_at_turn=0,
        stop_at_turn=1,
        stats=[
            'normal_mode_emittances',
            'covariance_optics',
            'gemitt_x_projected',
        ],
    )

    assert monitor.stats == (
        'gemitt_x', 'gemitt_y', 'gemitt_zeta',
        'nemitt_x', 'nemitt_y', 'nemitt_zeta',
        'betx', 'alfx', 'bety', 'alfy', 'betzeta', 'alfzeta',
        'dx', 'dpx', 'dy', 'dpy',
        'gemitt_x_projected',
    )

    monitor.data.num_particles[...] = weight
    monitor.data.sum_beta0_gamma0[...] = weight * beta0_gamma0
    coords = ('x', 'px', 'y', 'py', 'zeta', 'pzeta')
    storage_order = ('x', 'px', 'y', 'py', 'zeta', 'delta', 'pzeta')
    for ii, coord1 in enumerate(coords):
        for jj, coord2 in enumerate(coords[ii:], start=ii):
            if storage_order.index(coord1) <= storage_order.index(coord2):
                field_name = f'sum_{coord1}_{coord2}'
            else:
                field_name = f'sum_{coord2}_{coord1}'
            getattr(monitor.data, field_name)[...] = weight * sigma[ii, jj]

    assert_allclose(monitor.gemitt_x, [gemitt_x], rtol=1e-12, atol=0)
    assert_allclose(monitor.gemitt_y, [gemitt_y], rtol=1e-12, atol=0)
    assert_allclose(monitor.gemitt_zeta, [gemitt_zeta], rtol=1e-12, atol=0)
    assert_allclose(monitor.nemitt_x, [gemitt_x * beta0_gamma0],
                    rtol=1e-12, atol=0)
    assert_allclose(monitor.nemitt_y, [gemitt_y * beta0_gamma0],
                    rtol=1e-12, atol=0)
    assert_allclose(monitor.nemitt_zeta, [gemitt_zeta * beta0_gamma0],
                    rtol=1e-12, atol=0)
    assert_allclose(monitor.betx, [betx], rtol=1e-12, atol=0)
    assert_allclose(monitor.alfx, [alfx], rtol=1e-12, atol=0)
    assert_allclose(monitor.bety, [bety], rtol=1e-12, atol=0)
    assert_allclose(monitor.alfy, [alfy], rtol=1e-12, atol=0)
    assert_allclose(monitor.betzeta, [betzeta], rtol=1e-12, atol=0)
    assert_allclose(monitor.alfzeta, [alfzeta], rtol=1e-12, atol=0)
    assert_allclose(monitor.dx, [0.], atol=1e-14)
    assert_allclose(monitor.dpx, [0.], atol=1e-14)
    assert_allclose(monitor.dy, [0.], atol=1e-14)
    assert_allclose(monitor.dpy, [0.], atol=1e-14)

    out = monitor.optics_from_covariance(level='beam', turn=0)
    assert out['status'] == 'ok'
    assert out['covariance_order'] == coords
    assert_allclose(out['covariance_matrix'], sigma, rtol=1e-15, atol=0)
    assert_allclose(out['W_matrix'], w_matrix, rtol=1e-12, atol=1e-14)
    assert_allclose(out['gemitt_x'], gemitt_x, rtol=1e-12, atol=0)
    assert_allclose(out['nemitt_zeta'], gemitt_zeta * beta0_gamma0,
                    rtol=1e-12, atol=0)
    assert_allclose(out['betx'], betx, rtol=1e-12, atol=0)
    assert_allclose(out['alfzeta'], alfzeta, rtol=1e-12, atol=0)


@for_all_test_contexts
@allow_no_prebuilt_kernels
def test_beam_stats_monitor_covariance_optics_generated_matched_bunch(
        test_context):
    np.random.seed(12345)

    line = xt.Line(elements=[
        xt.LineSegmentMap(
            length=100.,
            betx=3., alfx=0.7, qx=0.31,
            bety=5., alfy=-0.4, qy=0.32,
            longitudinal_mode='linear_fixed_rf',
            voltage_rf=16e6,
            frequency_rf=400.8e6,
            phase_rf=np.pi,
            slippage_length=100.,
            momentum_compaction_factor=3.225e-4,
        ),
    ])
    line.set_particle_ref('proton', p0c=7e12)
    tw = line.twiss(method='6d')

    nemitt_x = 2.0e-6
    nemitt_y = 1.0e-6
    total_intensity = 1.0e11
    particles = xp.generate_matched_gaussian_bunch(
        num_particles=30_000,
        total_intensity_particles=total_intensity,
        nemitt_x=nemitt_x,
        nemitt_y=nemitt_y,
        sigma_z=0.08,
        line=line,
        engine='linear',
    )
    particles = particles.copy(_context=test_context)

    monitor = xt.BeamStatsMonitor(
        _context=test_context,
        start_at_turn=0,
        stop_at_turn=1,
        stats=[
            'num_particles',
            'normal_mode_emittances',
            'covariance_optics',
            'nemitt_x_projected',
            'nemitt_y_projected',
            'nemitt_zeta_projected',
        ],
    )
    monitor.track(particles)

    out = monitor.optics_from_covariance(level='beam', turn=0)
    assert out['status'] == 'ok'
    assert_allclose(monitor.num_particles, [total_intensity])

    # The generated bunch is uncoupled, so normal-mode and projected transverse
    # emittances should both recover the generation inputs up to finite-sample
    # noise.
    assert_allclose(monitor.nemitt_x, [nemitt_x], rtol=3e-2, atol=0)
    assert_allclose(monitor.nemitt_y, [nemitt_y], rtol=3e-2, atol=0)
    assert_allclose(
        monitor.nemitt_x_projected, [nemitt_x], rtol=3e-2, atol=0)
    assert_allclose(
        monitor.nemitt_y_projected, [nemitt_y], rtol=3e-2, atol=0)
    assert_allclose(
        monitor.nemitt_zeta, monitor.nemitt_zeta_projected,
        rtol=1e-3, atol=0)

    assert_allclose(monitor.betx, [tw.betx[0]], rtol=3e-2, atol=0)
    assert_allclose(monitor.alfx, [tw.alfx[0]], rtol=5e-2, atol=0)
    assert_allclose(monitor.bety, [tw.bety[0]], rtol=3e-2, atol=0)
    assert_allclose(monitor.alfy, [tw.alfy[0]], rtol=5e-2, atol=0)

    w_matrix = tw.W_matrix[0]
    betzeta = w_matrix[4, 4]**2 + w_matrix[4, 5]**2
    alfzeta = -(
        w_matrix[4, 4] * w_matrix[5, 4]
        + w_matrix[4, 5] * w_matrix[5, 5])
    assert_allclose(monitor.betzeta, [betzeta], rtol=3e-2, atol=0)
    assert_allclose(monitor.alfzeta, [alfzeta], atol=5e-2)

    assert_allclose(monitor.dx, [tw.dx[0]], atol=3e-4)
    assert_allclose(monitor.dpx, [tw.dpx[0]], atol=3e-4)
    assert_allclose(monitor.dy, [tw.dy[0]], atol=3e-4)
    assert_allclose(monitor.dpy, [tw.dpy[0]], atol=3e-4)

    assert_allclose(out['nemitt_x'], monitor.nemitt_x[0])
    assert_allclose(out['betx'], monitor.betx[0])


def test_beam_stats_monitor_optics_from_covariance_requires_moments():
    monitor = xt.BeamStatsMonitor(
        start_at_turn=0,
        stop_at_turn=1,
        stats=['num_particles'],
    )

    with pytest.raises(ValueError, match='Full 6D covariance moments'):
        monitor.optics_from_covariance(level='beam', turn=0)


@for_all_test_contexts
@allow_no_prebuilt_kernels
def test_beam_stats_monitor_selected_slots(test_context):
    particles = xt.Particles(
        _context=test_context,
        p0c=7e12,
        x=[1., 2., 10., 20.],
        px=[0., 0., 0., 0.],
        y=[0., 0., 0., 0.],
        py=[0., 0., 0., 0.],
        zeta=[-0.2, 0.2, -10.2, -9.8],
        delta=[0., 0., 0., 0.],
        weight=[1., 1., 3., 1.],
    )
    monitor = xt.BeamStatsMonitor(
        _context=test_context,
        start_at_turn=0,
        stop_at_turn=1,
        zeta_range=(-0.5, 0.5),
        num_slices=1,
        filled_slots=[0, 1],
        selected_slots=[1],
        bunch_spacing_zeta=10.,
        stats=['num_particles', 'mean_x'],
    )

    monitor.track(particles)

    assert_equal(monitor.selected_slots, [1])
    assert monitor.zeta_centers.shape == (1, 1)
    assert_allclose(monitor.zeta_centers, [[-10.]])
    assert_allclose(monitor.num_particles, [[[4.]]])
    assert_allclose(monitor.mean_x, [[[12.5]]])
    assert monitor.slot_index(1) == 0
    assert monitor.slice_index(-10., slot=1) == 0
    assert_allclose(monitor.get('mean_x', slot=1), [[12.5]])
    assert_allclose(monitor.get('mean_x', slot=1, slice_index=0), [12.5])


@for_all_test_contexts
@allow_no_prebuilt_kernels
def test_beam_stats_monitor_mixed_turns_and_lost_particle_zero(test_context):
    particles = xt.Particles(
        _context=test_context,
        p0c=7e12,
        x=[100., 1., 3., 10.],
        px=[0., 0., 0., 0.],
        y=[0., 0., 0., 0.],
        py=[0., 0., 0., 0.],
        zeta=[0., -0.5, -0.25, 0.5],
        delta=[0., 0., 0., 0.],
        weight=[100., 2., 1., 4.],
        state=[0, 1, 1, 1],
        at_turn=[0, 0, 0, 2],
    )
    monitor = xt.BeamStatsMonitor(
        _context=test_context,
        start_at_turn=0,
        stop_at_turn=3,
        zeta_range=(-1., 1.),
        num_slices=2,
        stats=['num_particles', 'mean_x'],
    )

    monitor.track(particles)

    assert_equal(monitor.turns, [0, 1, 2])
    assert_allclose(monitor.num_particles[:, 0, :],
                    [[3., 0.], [0., 0.], [0., 4.]])
    assert_allclose(monitor.mean_x[:, 0, :],
                    [[5. / 3., 0.], [0., 0.], [0., 10.]])
    assert_allclose(monitor.get('num_particles', level='beam'), [3., 0., 4.])
    assert_allclose(monitor.get('mean_x', level='beam'),
                    [5. / 3., 0., 10.])


def test_beam_stats_monitor_requires_spacing_for_non_default_slots():
    with pytest.raises(ValueError, match='bunch_spacing_zeta'):
        xt.BeamStatsMonitor(
            start_at_turn=0,
            stop_at_turn=1,
            filled_slots=[1],
            selected_slots=[1],
            stats=['num_particles'],
        )

    with pytest.raises(ValueError, match='bunch_spacing_zeta'):
        xt.BeamStatsMonitor(
            start_at_turn=0,
            stop_at_turn=1,
            filled_slots=[0, 1],
            selected_slots=[0],
            stats=['num_particles'],
        )


def test_beam_stats_monitor_rejects_duplicate_slots():
    with pytest.raises(ValueError, match='filled_slots.*duplicates'):
        xt.BeamStatsMonitor(
            start_at_turn=0,
            stop_at_turn=1,
            filled_slots=[0, 0],
            selected_slots=[0],
            bunch_spacing_zeta=10.,
            stats=['num_particles'],
        )

    with pytest.raises(ValueError, match='selected_slots.*duplicates'):
        xt.BeamStatsMonitor(
            start_at_turn=0,
            stop_at_turn=1,
            filled_slots=[0, 1],
            selected_slots=[1, 1],
            bunch_spacing_zeta=10.,
            stats=['num_particles'],
        )


@for_all_test_contexts
@allow_no_prebuilt_kernels
def test_beam_stats_monitor_coasting_slice_stats(test_context):
    monitor = xt.BeamStatsMonitor(
        _context=test_context,
        start_at_turn=0,
        stop_at_turn=3,
        coasting=True,
        num_slices=4,
        stats=['num_particles', 'mean_x', 'mean_zeta'],
    )
    line = xt.Line(elements=[monitor, xt.Drift(length=8.)])
    line.build_tracker(_context=test_context)

    particles = xt.Particles(
        _context=test_context,
        p0c=7e12,
        x=[1., 2., 3., 4., 5., 6.],
        px=[0., 0., 0., 0., 0., 0.],
        y=[0., 0., 0., 0., 0., 0.],
        py=[0., 0., 0., 0., 0., 0.],
        zeta=[3., 1., -1., -3., 9., -9.],
        delta=[0., 0., 0., 0., 0., 0.],
        weight=[1., 1., 1., 1., 1., 1.],
        at_turn=[1, 1, 1, 1, 1, 1],
    )

    line.track(particles, num_turns=1)

    assert monitor.coasting
    assert monitor.available_levels == ('beam', 'slice')
    assert monitor.default_level == 'slice'
    assert_equal(monitor.selected_slots, [0])
    assert_equal(monitor.filled_slots, [0])
    assert monitor.zeta_centers is None
    assert monitor.num_particles.shape == (3, 4)
    assert monitor.data.num_particles.shape == (12,)

    assert_allclose(
        monitor.num_particles,
        [[0., 1., 0., 0.],
         [1., 1., 1., 1.],
         [0., 0., 1., 0.]])
    assert_allclose(
        monitor.mean_x,
        [[0., 5., 0., 0.],
         [1., 2., 3., 4.],
         [0., 0., 6., 0.]])
    assert_allclose(
        monitor.mean_zeta,
        [[0., 1., 0., 0.],
         [3., 1., -1., -3.],
         [0., 0., -1., 0.]])
    assert_allclose(
        monitor.get('num_particles', level='beam'),
        [1., 4., 1.])
    assert_allclose(
        monitor.get('mean_x', level='beam'),
        [5., 2.5, 6.])
    assert monitor.slice_index(9., line_length=8.) == 1
    assert monitor.slice_index(-9., line_length=8.) == 2

    assert_allclose(
        monitor.zeta_centers_unwrapped(line_length=8.),
        [[3., 1., -1., -3.],
         [-5., -7., -9., -11.],
         [-13., -15., -17., -19.]])
    assert_allclose(
        monitor.time_centers(line_length=8., beta0=1.) * 299792458.,
        [[-3., -1., 1., 3.],
         [5., 7., 9., 11.],
         [13., 15., 17., 19.]])
    assert_allclose(monitor.get('mean_x', turn=1), [1., 2., 3., 4.])
    assert_allclose(monitor.get('mean_x', turn=1, slice_index=2), 3.)
    assert monitor.get('mean_x', turn=1, slice_index=2,
                       keepdims=True).shape == (1, 1)
    with pytest.raises(ValueError, match='level.*beam.*slice'):
        monitor.get('mean_x', level='bunch')
    with pytest.raises(ValueError, match='slot.*coasting'):
        monitor.get('mean_x', slot=0)
    with pytest.raises(ValueError, match='slot.*coasting'):
        monitor.slice_index(9., slot=0, line_length=8.)


@for_all_test_contexts
@allow_no_prebuilt_kernels
def test_beam_stats_monitor_coasting_hdf5_public_shape(test_context, tmp_path):
    h5py = pytest.importorskip('h5py')

    output_file = tmp_path / 'beam_stats_monitor_coasting.h5'
    monitor = xt.BeamStatsMonitor(
        _context=test_context,
        start_at_turn=0,
        stop_at_turn=3,
        coasting=True,
        num_slices=4,
        stats=['num_particles', 'mean_x'],
        output_file=output_file,
    )
    line = xt.Line(elements=[monitor, xt.Drift(length=8.)])
    line.build_tracker(_context=test_context)

    particles = xt.Particles(
        _context=test_context,
        p0c=7e12,
        x=[1., 2., 3., 4., 5., 6.],
        zeta=[3., 1., -1., -3., 9., -9.],
        weight=[1., 1., 1., 1., 1., 1.],
        at_turn=[1, 1, 1, 1, 1, 1],
    )

    line.track(particles, num_turns=1)
    monitor.save_to_file()

    with h5py.File(output_file, 'r') as h5file:
        assert bool(h5file.attrs['coasting'])
        assert_equal(h5file.attrs['available_levels'].astype(str),
                     ['beam', 'slice'])
        assert h5file.attrs['default_level'] == 'slice'
        assert 'bunch' not in h5file['stats']
        assert 'filled_slots' not in h5file
        assert 'selected_slots' not in h5file
        assert_allclose(h5file['stats/slice/num_particles'][...],
                        monitor.num_particles)
        assert h5file['stats/slice/num_particles'].shape == (3, 4)
        assert_allclose(h5file['stats/beam/num_particles'][...],
                        [1., 4., 1.])


def test_beam_stats_monitor_coasting_rejects_bunched_inputs():
    with pytest.raises(ValueError, match='zeta_range'):
        xt.BeamStatsMonitor(
            start_at_turn=0,
            stop_at_turn=1,
            coasting=True,
            zeta_range=(-1., 1.),
            num_slices=4,
            stats=['num_particles'],
        )

    with pytest.raises(ValueError, match='Bunched-beam filling inputs'):
        xt.BeamStatsMonitor(
            start_at_turn=0,
            stop_at_turn=1,
            coasting=True,
            num_slices=4,
            selected_slots=[0],
            stats=['num_particles'],
        )

    with pytest.raises(ValueError, match='num_slices'):
        xt.BeamStatsMonitor(
            start_at_turn=0,
            stop_at_turn=1,
            coasting=True,
            stats=['num_particles'],
        )


def test_beam_stats_monitor_coasting_to_dict_stores_configuration_only():
    monitor = xt.BeamStatsMonitor(
        start_at_turn=3,
        stop_at_turn=7,
        every_n_turns=2,
        coasting=True,
        num_slices=5,
        stats=['num_particles', 'mean_x'],
    )

    assert monitor.to_dict() == {
        '__class__': 'BeamStatsMonitor',
        'start_at_turn': 3,
        'stop_at_turn': 7,
        'every_n_turns': 2,
        'stats': ['num_particles', 'mean_x'],
        'coasting': True,
        'num_slices': 5,
    }

    line = xt.Line(elements=[monitor])
    line_from_dict = xt.Line.from_dict(line.to_dict())
    monitor_from_dict = line_from_dict['e0']
    assert monitor_from_dict.coasting
    assert monitor_from_dict.available_levels == ('beam', 'slice')
    assert monitor_from_dict.num_particles.shape == (2, 5)
    assert_equal(monitor_from_dict.selected_slots, [0])
    assert monitor_from_dict.zeta_centers is None


def test_beam_stats_monitor_coasting_rejects_start_new_frame():
    monitor = xt.BeamStatsMonitor(
        start_at_turn=3,
        stop_at_turn=7,
        every_n_turns=2,
        coasting=True,
        num_slices=5,
        stats=['num_particles'],
    )

    with pytest.raises(ValueError, match='start_new_frame.*coasting'):
        monitor.start_new_frame(start_at_turn=9)

    assert_equal(monitor.turns, [3, 5])


@for_all_test_contexts
@allow_no_prebuilt_kernels
def test_beam_stats_monitor_profiles_whole_beam(test_context):
    particles = xt.Particles(
        _context=test_context,
        p0c=7e12,
        x=[-0.75, -0.25, 0.25, 0.75, 1.25],
        y=[0., 0., 0., 0., 0.],
        weight=[1., 2., 3., 4., 5.],
    )
    monitor = xt.BeamStatsMonitor(
        _context=test_context,
        start_at_turn=0,
        stop_at_turn=2,
        stats=['num_particles'],
        profiles={
            'x': {'range': (-1., 1.), 'num_bins': 4},
            'y': {'range': (-1., 1.), 'num_bins': 2},
        },
    )

    monitor.track(particles)
    particles.at_turn += 1
    particles.x = [-0.75, -0.25, 0.25, 0.75, 1.25]
    particles.weight = [5., 4., 3., 2., 1.]
    monitor.track(particles)

    assert monitor.profile_coordinates == ('x', 'y')
    assert monitor.profile_num_bins == {'x': 4, 'y': 2}
    assert_allclose(monitor.profile_bin_edges['x'],
                    [-1., -0.5, 0., 0.5, 1.])
    assert_allclose(monitor.profile_bin_centers['x'],
                    [-0.75, -0.25, 0.25, 0.75])
    assert_allclose(monitor.profiles['x'],
                    [[1., 2., 3., 4.],
                     [5., 4., 3., 2.]])
    assert_allclose(monitor.profiles['y'],
                    [[0., 15.],
                     [0., 15.]])
    assert_allclose(monitor.num_particles, [15., 15.])


@for_all_test_contexts
@allow_no_prebuilt_kernels
def test_beam_stats_monitor_profiles_slice_and_coasting_shapes(test_context):
    slice_particles = xt.Particles(
        _context=test_context,
        p0c=7e12,
        x=[-0.75, -0.25, 0.25, 0.75],
        zeta=[-0.75, -0.25, 0.25, 0.75],
        weight=[1., 2., 3., 4.],
    )
    slice_monitor = xt.BeamStatsMonitor(
        _context=test_context,
        start_at_turn=0,
        stop_at_turn=1,
        zeta_range=(-1., 1.),
        num_slices=2,
        stats=['num_particles'],
        profiles={'x': {'range': (-1., 1.), 'num_bins': 4}},
    )

    slice_monitor.track(slice_particles)

    assert slice_monitor.profiles['x'].shape == (1, 1, 2, 4)
    assert_allclose(slice_monitor.profiles['x'][0, 0],
                    [[1., 2., 0., 0.],
                     [0., 0., 3., 4.]])

    coasting_monitor = xt.BeamStatsMonitor(
        _context=test_context,
        start_at_turn=0,
        stop_at_turn=3,
        coasting=True,
        num_slices=4,
        stats=['num_particles'],
        profiles={
            'x': {'range': (0., 8.), 'num_bins': 4},
            'zeta': {'range': (-4., 4.), 'num_bins': 4},
        },
    )
    line = xt.Line(elements=[coasting_monitor, xt.Drift(length=8.)])
    line.build_tracker(_context=test_context)
    coasting_particles = xt.Particles(
        _context=test_context,
        p0c=7e12,
        x=[1., 3., 5., 7.],
        zeta=[3., 1., -1., -3.],
        weight=[1., 2., 3., 4.],
        at_turn=[1, 1, 1, 1],
    )

    line.track(coasting_particles, num_turns=1)

    assert coasting_monitor.profiles['x'].shape == (3, 4, 4)
    assert_allclose(
        coasting_monitor.profiles['x'],
        np.array([
            [[0., 0., 0., 0.],
             [0., 0., 0., 0.],
             [0., 0., 0., 0.],
             [0., 0., 0., 0.]],
            [[1., 0., 0., 0.],
             [0., 2., 0., 0.],
             [0., 0., 3., 0.],
             [0., 0., 0., 4.]],
            [[0., 0., 0., 0.],
             [0., 0., 0., 0.],
             [0., 0., 0., 0.],
             [0., 0., 0., 0.]],
        ]))
    assert_allclose(
        coasting_monitor.profiles['zeta'][1],
        [[0., 0., 0., 1.],
         [0., 0., 2., 0.],
         [0., 3., 0., 0.],
         [4., 0., 0., 0.]])


def test_beam_stats_monitor_profiles_validation():
    with pytest.raises(ValueError, match='Unknown coordinate'):
        xt.BeamStatsMonitor(
            start_at_turn=0,
            stop_at_turn=1,
            profiles={'not_a_coord': {'range': (-1., 1.), 'num_bins': 4}},
        )

    with pytest.raises(ValueError, match='missing `range`'):
        xt.BeamStatsMonitor(
            start_at_turn=0,
            stop_at_turn=1,
            profiles={'x': {'num_bins': 4}},
        )

    with pytest.raises(ValueError, match='num_bins.*positive'):
        xt.BeamStatsMonitor(
            start_at_turn=0,
            stop_at_turn=1,
            profiles={'x': {'range': (-1., 1.), 'num_bins': 0}},
        )

    with pytest.raises(ValueError, match='range.*increasing'):
        xt.BeamStatsMonitor(
            start_at_turn=0,
            stop_at_turn=1,
            profiles={'x': {'range': (1., -1.), 'num_bins': 4}},
        )

    with pytest.raises(ValueError, match='unsupported keys'):
        xt.BeamStatsMonitor(
            start_at_turn=0,
            stop_at_turn=1,
            profiles={
                'x': {
                    'coordinate': 'x',
                    'range': (-1., 1.),
                    'num_bins': 4,
                },
            },
        )


@for_all_test_contexts
@allow_no_prebuilt_kernels
def test_beam_stats_monitor_reset_data(test_context):
    particles = xt.Particles(
        _context=test_context,
        p0c=7e12,
        x=[1., 3.],
        px=[0., 0.],
        y=[0., 0.],
        py=[0., 0.],
        zeta=[0., 0.],
        delta=[0., 0.],
        weight=[2., 1.],
    )
    monitor = xt.BeamStatsMonitor(
        _context=test_context,
        start_at_turn=0,
        stop_at_turn=1,
        stats=['num_particles', 'mean_x', 'sigma_x'],
        profiles={'x': {'range': (0., 4.), 'num_bins': 2}},
    )

    monitor.track(particles)

    assert_allclose(monitor.num_particles, [3.])
    assert_allclose(monitor.mean_x, [5. / 3.])
    assert_allclose(monitor.profiles['x'], [[2., 1.]])
    if isinstance(test_context, xo.ContextCpu):
        assert isinstance(monitor.data.num_particles, np.ndarray)

    monitor._reset_data()

    assert_allclose(monitor.num_particles, [0.])
    assert_allclose(monitor.mean_x, [0.])
    assert_allclose(monitor.profiles['x'], [[0., 0.]])
    for field in monitor._RAW_FIELDS:
        assert_allclose(_to_numpy(test_context, getattr(monitor.data, field)),
                        0.)


@for_all_test_contexts
@allow_no_prebuilt_kernels
def test_beam_stats_monitor_save_to_file_hdf5(test_context, tmp_path):
    h5py = pytest.importorskip('h5py')

    particles = xt.Particles(
        _context=test_context,
        p0c=7e12,
        x=[1., 3., 10.],
        px=[0.1, 0.3, 1.0],
        y=[2., 4., 20.],
        py=[0.2, 0.4, 2.0],
        zeta=[-0.5, -0.25, 0.5],
        delta=[0.01, 0.03, 0.1],
        weight=[2., 1., 4.],
    )
    output_file = tmp_path / 'beam_stats_monitor.h5'
    monitor = xt.BeamStatsMonitor(
        _context=test_context,
        start_at_turn=0,
        stop_at_turn=1,
        zeta_range=(-1., 1.),
        num_slices=2,
        stats=['num_particles', 'mean_x', 'sigma_x'],
        profiles={'x': {'range': (0., 12.), 'num_bins': 3}},
        output_file=output_file,
    )

    monitor.track(particles)
    monitor.save_to_file()
    monitor.save_to_file()

    with h5py.File(output_file, 'r') as h5file:
        assert h5file.attrs['schema_version'] == 1
        assert h5file.attrs['class'] == 'BeamStatsMonitor'
        assert h5file.attrs['default_level'] == 'slice'
        assert h5file.attrs['n_records_per_frame'] == 1
        assert_equal(h5file.attrs['stats'].astype(str),
                     ['num_particles', 'mean_x', 'sigma_x'])
        assert_equal(h5file.attrs['available_levels'].astype(str),
                     ['beam', 'bunch', 'slice'])
        assert_equal(h5file['turns'][...], [0])
        assert 'frames' not in h5file
        assert_equal(h5file['selected_slots'][...], [0])
        assert_allclose(h5file['zeta_centers'][...], [[-0.5, 0.5]])
        assert 'moments' not in h5file
        assert_allclose(h5file['stats/slice/mean_x'][...],
                        monitor.get('mean_x', level='slice'))
        assert_allclose(h5file['stats/bunch/sigma_x'][...],
                        monitor.get('sigma_x', level='bunch'))
        assert_allclose(h5file['stats/beam/num_particles'][...],
                        monitor.get('num_particles', level='beam'))
        assert_equal(h5file.attrs['profile_coordinates'].astype(str), ['x'])
        assert_allclose(h5file['profiles/x/bin_edges'][...],
                        [0., 4., 8., 12.])
        assert_allclose(h5file['profiles/x/bin_centers'][...],
                        [2., 6., 10.])
        assert_allclose(h5file['profiles/x/counts'][...],
                        monitor.profiles['x'])


def test_beam_stats_monitor_output_file_is_initialized_on_creation(tmp_path):
    h5py = pytest.importorskip('h5py')

    output_file = tmp_path / 'beam_stats_monitor_existing.h5'
    with h5py.File(output_file, 'w') as h5file:
        h5file.create_dataset('old_data', data=[1, 2, 3])

    xt.BeamStatsMonitor(
        start_at_turn=0,
        stop_at_turn=2,
        stats=['num_particles', 'mean_x'],
        output_file=output_file,
    )

    with h5py.File(output_file, 'r') as h5file:
        assert 'old_data' not in h5file
        assert h5file.attrs['class'] == 'BeamStatsMonitor'
        assert_equal(h5file.attrs['stats'].astype(str),
                     ['num_particles', 'mean_x'])
        assert_equal(h5file['filled_slots'][...], [])
        assert_equal(h5file['selected_slots'][...], [])
        assert 'turns' not in h5file
        assert 'stats' not in h5file


@for_all_test_contexts
@allow_no_prebuilt_kernels
def test_beam_stats_monitor_save_to_file_creates_filename(
        test_context, tmp_path):
    h5py = pytest.importorskip('h5py')

    output_file = tmp_path / 'beam_stats_monitor_save_later.h5'

    particles = xt.Particles(
        _context=test_context,
        p0c=7e12,
        x=[1., 3.],
        px=[0., 0.],
        y=[0., 0.],
        py=[0., 0.],
        zeta=[0., 0.],
        delta=[0., 0.],
        weight=[1., 1.],
    )
    monitor = xt.BeamStatsMonitor(
        _context=test_context,
        start_at_turn=0,
        stop_at_turn=2,
        stats=['num_particles', 'mean_x'],
    )
    monitor.track(particles)

    monitor.save_to_file(output_file)
    monitor.save_to_file(output_file)

    with h5py.File(output_file, 'r') as h5file:
        assert_equal(h5file['turns'][...], [0])
        assert_allclose(h5file['stats/beam/num_particles'][...], [2.])
        assert_allclose(h5file['stats/beam/mean_x'][...], [2.])


@for_all_test_contexts
@allow_no_prebuilt_kernels
def test_beam_stats_monitor_save_to_file_appends_existing_filename(
        test_context, tmp_path):
    h5py = pytest.importorskip('h5py')

    output_file = tmp_path / 'beam_stats_monitor_append_later.h5'
    particles = xt.Particles(
        _context=test_context,
        p0c=7e12,
        x=[1., 3.],
        px=[0., 0.],
        y=[0., 0.],
        py=[0., 0.],
        zeta=[0., 0.],
        delta=[0., 0.],
        weight=[1., 1.],
    )

    first_monitor = xt.BeamStatsMonitor(
        _context=test_context,
        start_at_turn=0,
        stop_at_turn=2,
        stats=['num_particles', 'mean_x'],
        output_file=output_file,
    )
    first_monitor.track(particles)
    first_monitor.save_to_file()

    second_monitor = xt.BeamStatsMonitor(
        _context=test_context,
        start_at_turn=0,
        stop_at_turn=2,
        stats=['num_particles', 'mean_x'],
    )
    particles.at_turn = 0
    second_monitor.track(particles)
    particles.at_turn += 1
    particles.x += 2
    second_monitor.track(particles)

    second_monitor.save_to_file(output_file)
    second_monitor.save_to_file(output_file)

    with h5py.File(output_file, 'r') as h5file:
        assert_equal(h5file['turns'][...], [0, 1])
        assert_allclose(h5file['stats/beam/num_particles'][...], [2., 2.])
        assert_allclose(h5file['stats/beam/mean_x'][...], [2., 4.])


@for_all_test_contexts
@allow_no_prebuilt_kernels
def test_beam_stats_monitor_save_to_file_rejects_invalid_existing_file(
        test_context, tmp_path):
    h5py = pytest.importorskip('h5py')

    output_file = tmp_path / 'beam_stats_monitor_invalid.h5'
    with h5py.File(output_file, 'w') as h5file:
        h5file.create_dataset('old_data', data=[1, 2, 3])

    particles = xt.Particles(
        _context=test_context,
        p0c=7e12,
        x=[1., 3.],
        px=[0., 0.],
        y=[0., 0.],
        py=[0., 0.],
        zeta=[0., 0.],
        delta=[0., 0.],
        weight=[1., 1.],
    )
    monitor = xt.BeamStatsMonitor(
        _context=test_context,
        start_at_turn=0,
        stop_at_turn=2,
        stats=['num_particles', 'mean_x'],
    )
    monitor.track(particles)

    with pytest.raises(ValueError, match='not empty'):
        monitor.save_to_file(output_file)

    with h5py.File(output_file, 'r') as h5file:
        assert 'old_data' in h5file


@for_all_test_contexts
@allow_no_prebuilt_kernels
def test_beam_stats_monitor_hdf5_progressive_save_to_file(
        test_context, tmp_path):
    h5py = pytest.importorskip('h5py')

    output_file = tmp_path / 'beam_stats_monitor_progress.h5'
    monitor = xt.BeamStatsMonitor(
        _context=test_context,
        start_at_turn=0,
        stop_at_turn=3,
        stats=['num_particles', 'mean_x'],
        output_file=output_file,
    )

    particles = xt.Particles(
        _context=test_context,
        p0c=7e12,
        x=[1., 3.],
        px=[0., 0.],
        y=[0., 0.],
        py=[0., 0.],
        zeta=[0., 0.],
        delta=[0., 0.],
        weight=[1., 1.],
    )

    monitor.track(particles)
    monitor.save_to_file()

    with h5py.File(output_file, 'r') as h5file:
        assert_equal(h5file['turns'][...], [0])
        assert_allclose(h5file['stats/beam/num_particles'][...], [2.])
        assert_allclose(h5file['stats/beam/mean_x'][...], [2.])

    particles.x = [10., 20.]
    particles.at_turn += 1
    monitor.track(particles)
    monitor.save_to_file()
    monitor.save_to_file()

    with h5py.File(output_file, 'r') as h5file:
        assert_equal(h5file['turns'][...], [0, 1])
        assert_allclose(h5file['stats/beam/num_particles'][...], [2., 2.])
        assert_allclose(h5file['stats/beam/mean_x'][...], [2., 15.])
        assert 'frames' not in h5file
        assert 'moments' not in h5file


@for_all_test_contexts
@allow_no_prebuilt_kernels
def test_beam_stats_monitor_hdf5_start_new_frame(test_context, tmp_path):
    h5py = pytest.importorskip('h5py')

    output_file = tmp_path / 'beam_stats_monitor_new_frame.h5'
    monitor = xt.BeamStatsMonitor(
        _context=test_context,
        start_at_turn=0,
        stop_at_turn=1,
        stats=['num_particles', 'mean_x'],
        output_file=output_file,
    )

    particles = xt.Particles(
        _context=test_context,
        p0c=7e12,
        x=[1., 3.],
        px=[0., 0.],
        y=[0., 0.],
        py=[0., 0.],
        zeta=[0., 0.],
        delta=[0., 0.],
        weight=[1., 1.],
    )
    monitor.track(particles)
    monitor.save_to_file()

    monitor.start_new_frame(start_at_turn=1)
    assert_equal(monitor.turns, [1])
    assert_allclose(monitor.num_particles, [0.])

    particles = xt.Particles(
        _context=test_context,
        p0c=7e12,
        x=[10., 20.],
        px=[0., 0.],
        y=[0., 0.],
        py=[0., 0.],
        zeta=[0., 0.],
        delta=[0., 0.],
        weight=[1., 1.],
        at_turn=[1, 1],
    )
    monitor.track(particles)
    monitor.save_to_file()

    with h5py.File(output_file, 'r') as h5file:
        assert_equal(h5file['turns'][...], [0, 1])
        assert_allclose(h5file['stats/beam/num_particles'][...], [2., 2.])
        assert_allclose(h5file['stats/beam/mean_x'][...], [2., 15.])
        assert 'frames' not in h5file


@for_all_test_contexts
@allow_no_prebuilt_kernels
def test_beam_stats_monitor_to_dict_stores_configuration_only(test_context):
    particles = xt.Particles(
        _context=test_context,
        p0c=7e12,
        x=[1., 2.],
        px=[0., 0.],
        y=[0., 0.],
        py=[0., 0.],
        zeta=[-0.2, -10.2],
        delta=[0., 0.],
        weight=[1., 1.],
    )
    monitor = xt.BeamStatsMonitor(
        _context=test_context,
        start_at_turn=0,
        stop_at_turn=2,
        every_n_turns=1,
        zeta_range=(-0.5, 0.5),
        num_slices=2,
        filled_slots=[0, 1],
        selected_slots=[1],
        bunch_spacing_zeta=10.,
        stats=['num_particles', 'mean_x'],
        profiles={'x': {'range': (-1., 1.), 'num_bins': 4}},
    )
    line = xt.Line(elements=[monitor])
    line.build_tracker(_context=test_context)

    line.track(particles, num_turns=1)
    assert_allclose(monitor.num_particles, [[[1., 0.]], [[0., 0.]]])

    monitor_dict = monitor.to_dict()
    assert 'data' not in monitor_dict
    assert monitor_dict == {
        '__class__': 'BeamStatsMonitor',
        'start_at_turn': 0,
        'stop_at_turn': 2,
        'every_n_turns': 1,
        'stats': ['num_particles', 'mean_x'],
        'zeta_range': (-0.5, 0.5),
        'num_slices': 2,
        'filled_slots': [0, 1],
        'selected_slots': [1],
        'bunch_spacing_zeta': 10.0,
        'profiles': {'x': {'range': (-1.0, 1.0), 'num_bins': 4}},
    }
    assert 'data' not in line.to_dict()['elements']['e0']

    line_from_dict = xt.Line.from_dict(line.to_dict())
    monitor_from_dict = line_from_dict['e0']
    assert monitor_from_dict.stats == ('num_particles', 'mean_x')
    assert monitor_from_dict.available_levels == ('beam', 'bunch', 'slice')
    assert_equal(monitor_from_dict.selected_slots, [1])
    assert_allclose(monitor_from_dict.zeta_centers, [[-10.25, -9.75]])
    assert_allclose(monitor_from_dict.num_particles,
                    np.zeros((2, 1, 2)))
    assert monitor_from_dict.profile_coordinates == ('x',)
    assert_allclose(monitor_from_dict.profiles['x'],
                    np.zeros((2, 1, 2, 4)))
