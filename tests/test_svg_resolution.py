"""Check the SVG approximation before aperture polygon resampling."""

import numpy as np
import pytest

from xtrack.svgutils import svg_to_points_by_resolution


CIRCLE_PATH = 'M1,0 A1,1 0 0,1 -1,0 A1,1 0 0,1 1,0 Z'


def test_straight_edges_keep_endpoints_only():
    """Straight edges need no interior samples; repeated vertices are removed.

    The expected coordinates also check the default mm-to-m scale, SVG y-axis
    inversion, and closure by returning to the start without a Z command.
    """
    path = 'M0,0 L0,0 L2,0 L2,1 L0,0'

    points = svg_to_points_by_resolution(path, resolution=1e-4)

    np.testing.assert_array_equal(points, [
        [0, 0],
        [0.002, 0],
        [0.002, -0.001],
        [0, 0],
    ])


def test_circle_sampling_uses_physical_segment_length():
    """A scaled circle gets the point count implied by length/resolution.

    Circular arcs have uniform speed in their parameter, so their samples
    must also have equal angular spacing and stay on the scaled circle.
    """
    scale = 0.002
    resolution = 0.0002
    intervals_per_half_circle = int(np.ceil(np.pi * scale / resolution))

    points = svg_to_points_by_resolution(CIRCLE_PATH, resolution, scale=scale)
    angles = np.unwrap(np.arctan2(-points[:, 1], points[:, 0]))

    assert len(points) == 2 * intervals_per_half_circle + 1

    np.testing.assert_allclose(np.linalg.norm(points, axis=1), scale, atol=1e-16)
    np.testing.assert_allclose(
        np.diff(angles), np.pi / intervals_per_half_circle, atol=1e-14,
    )
    np.testing.assert_array_equal(points[-1], points[0])


def test_short_quadratic_curve_keeps_its_midpoint():
    """Even a curve shorter than resolution needs an interior sample.

    Otherwise this quadratic and its closing line would collapse into a
    zero-area polygon. Its midpoint is known analytically.
    """
    points = svg_to_points_by_resolution(
        'M0,0 Q1,2 2,0 Z', resolution=100, scale=1,
    )

    np.testing.assert_array_equal(points, [[0, 0], [1, -1], [2, 0], [0, 0]])


def test_closed_cubic_is_not_mistaken_for_a_zero_length_segment():
    """Coincident endpoints do not make a curved segment degenerate."""
    points = svg_to_points_by_resolution(
        'M0,0 C2,0 0,2 0,0 Z', resolution=0.1, scale=1,
    )

    assert len(points) > 4
    assert np.max(np.linalg.norm(points, axis=1)) > 0.5

    np.testing.assert_array_equal(points[-1], points[0])


@pytest.mark.parametrize('path, message', [
    pytest.param('M0,0 L1,0 L1,1', 'closed', id='open-path'),
    pytest.param(
        'M0,0 L1,0 L1,1 Z M2,2 L3,2 L3,3 Z',
        'one subpath', id='multiple-subpaths',
    ),
    pytest.param('M0,0 L1,0 L2,0 Z', 'nonzero area', id='collinear-boundary'),
    pytest.param('M0,0 L1e999,0 L1,1 Z', 'finite', id='non-finite-coordinate'),
])
def test_reject_paths_that_cannot_describe_an_aperture(path, message):
    """Require a finite, closed boundary consisting of a single subpath."""
    with pytest.raises(ValueError, match=message):
        svg_to_points_by_resolution(path, resolution=0.1)


@pytest.mark.parametrize('scale, resolution, message', [
    pytest.param(0, 0.1, 'scale', id='zero-scale'),
    pytest.param(1, np.nan, 'resolution', id='non-finite-resolution'),
])
def test_reject_invalid_sampling_parameters(scale, resolution, message):
    """Reject invalid units or sampling distances before evaluating the path."""
    with pytest.raises(ValueError, match=message):
        svg_to_points_by_resolution(CIRCLE_PATH, resolution, scale=scale)
