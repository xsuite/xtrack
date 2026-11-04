import numpy as np
import pytest

from xtrack.svgutils import svg_to_points, svg_to_points_by_resolution


RECT = 'M-2,-1 L2,-1 L2,1 L-2,1 Z'
CIRCLE = 'M1,0 A1,1 0 0,1 -1,0 A1,1 0 0,1 1,0 Z'


def test_straight_segments_use_only_endpoints():
    expected = [[-2, 1], [2, 1], [2, -1], [-2, -1], [-2, 1]]
    for resolution in [1e-6, 1, 100]:
        np.testing.assert_array_equal(
            svg_to_points_by_resolution(RECT, resolution, scale=1), expected)


@pytest.mark.parametrize('scale', [0.001, 0.002])
def test_circle_density_scale_and_boundary(scale):
    resolution = 0.0002
    points = svg_to_points_by_resolution(CIRCLE, resolution, scale=scale)
    intervals_per_arc = int(np.ceil(np.pi * scale / resolution))
    assert len(points) == 2 * intervals_per_arc + 1
    np.testing.assert_allclose(np.linalg.norm(points, axis=1), scale, atol=1e-16)
    angles = np.unwrap(np.arctan2(-points[:, 1], points[:, 0]))
    np.testing.assert_allclose(np.diff(angles), np.pi / intervals_per_arc, atol=1e-14)
    np.testing.assert_array_equal(points[-1], points[0])


def test_elliptical_arc_boundary_and_shared_endpoint():
    points = svg_to_points_by_resolution('M2,0 A2,1 0 0,1 -2,0 Z', 0.1, scale=1)
    np.testing.assert_allclose((points[:-1, 0] / 2)**2 + points[:-1, 1]**2, 1, atol=1e-14)
    np.testing.assert_array_equal(points[-2:], [[-2, 0], [2, 0]])
    assert np.all(points[:-1, 1] <= 0)
    assert np.all(np.linalg.norm(np.diff(points, axis=0), axis=1) > 0)


def test_quadratic_and_cubic_samples():
    quadratic = svg_to_points_by_resolution('M0,0 Q1,2 2,0 Z', resolution=1, scale=1)
    np.testing.assert_allclose(quadratic, [
        [0, 0], [2/3, -8/9], [4/3, -8/9], [2, 0], [0, 0]])
    cubic = svg_to_points_by_resolution('M0,0 C0,3 2,-1 2,0 Z', resolution=100, scale=1)
    np.testing.assert_allclose(cubic, [[0, 0], [1, -0.75], [2, 0], [0, 0]])


def test_short_curve_has_interior_sample():
    points = svg_to_points_by_resolution('M0,0 Q1,2 2,0 Z', resolution=100, scale=1)
    np.testing.assert_array_equal(points, [[0, 0], [1, -1], [2, 0], [0, 0]])


def test_duplicates_and_explicit_endpoint_closure():
    path = 'M0,0 L0,0 L2,0 L2,1 L0,0'
    points = svg_to_points_by_resolution(path, resolution=1e-6, scale=1)
    np.testing.assert_array_equal(points, [[0, 0], [2, 0], [2, -1], [0, 0]])


def test_curve_returning_to_start_is_not_dropped():
    points = svg_to_points_by_resolution(
        'M0,0 C2,0 0,2 0,0 Z', resolution=0.1, scale=1)
    assert len(points) > 4
    assert np.max(np.linalg.norm(points, axis=1)) > 0.5
    np.testing.assert_array_equal(points[-1], points[0])


@pytest.mark.parametrize('path, message', [
    ('', 'non-empty'),
    ('M0,0 L1,0 L1,1', 'closed'),
    ('M0,0 L1,0 L1,1 Z M2,2 L3,2 L3,3 Z', 'one subpath'),
    ('M0,0 Z', 'non-degenerate'),
    ('M0,0 L1,0 L2,0 Z', 'nonzero area'),
    ('M0,0 L1e999,0 L1,1 Z', 'finite'),
])
def test_invalid_paths(path, message):
    with pytest.raises(ValueError, match=message):
        svg_to_points_by_resolution(path, resolution=1e-6)


@pytest.mark.parametrize('parameter', ['scale', 'resolution'])
@pytest.mark.parametrize('value', [0, -1, np.nan, np.inf])
def test_invalid_scale_and_resolution(parameter, value):
    kwargs = {'scale': 1, 'resolution': 0.1, parameter: value}
    with pytest.raises(ValueError, match=parameter):
        svg_to_points_by_resolution(RECT, **kwargs)


def test_smaller_resolution_improves_circle_approximation():
    errors = []
    for resolution in [0.5, 0.1, 0.02]:
        points = svg_to_points_by_resolution(CIRCLE, resolution=resolution, scale=1)
        midpoints = (points[1:] + points[:-1]) / 2
        errors.append(np.max(1 - np.linalg.norm(midpoints, axis=1)))
    assert errors[2] < errors[1] < errors[0]


def test_old_sampler_unchanged():
    # Keep the old stepping convention, including its historical skipped endpoint.
    xs, ys = svg_to_points('M0,0 L2,0 L2,2 Z', scale=1)
    np.testing.assert_array_equal(xs, [0, 2, 2])
    np.testing.assert_array_equal(ys, [0, 0, -2])
