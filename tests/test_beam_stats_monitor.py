import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

from xobjects.test_helpers import allow_no_prebuilt_kernels

import xtrack as xt


@allow_no_prebuilt_kernels
def test_beam_stats_monitor_whole_beam_stats():
    particles = xt.Particles(
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


@allow_no_prebuilt_kernels
def test_beam_stats_monitor_bunch_stats():
    particles = xt.Particles(
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


@allow_no_prebuilt_kernels
def test_beam_stats_monitor_weighted_stats_and_turn_selection():
    particles = xt.Particles(
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


@allow_no_prebuilt_kernels
def test_beam_stats_monitor_selected_slots():
    particles = xt.Particles(
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


@allow_no_prebuilt_kernels
def test_beam_stats_monitor_mixed_turns_and_lost_particle_zero():
    particles = xt.Particles(
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


@allow_no_prebuilt_kernels
def test_beam_stats_monitor_to_dict_stores_configuration_only():
    particles = xt.Particles(
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
        start_at_turn=0,
        stop_at_turn=2,
        every_n_turns=1,
        zeta_range=(-0.5, 0.5),
        num_slices=2,
        filled_slots=[0, 1],
        selected_slots=[1],
        bunch_spacing_zeta=10.,
        stats=['num_particles', 'mean_x'],
    )
    line = xt.Line(elements=[monitor])

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
