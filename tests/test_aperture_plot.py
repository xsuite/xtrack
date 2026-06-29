import numpy as np
import pytest

from xtrack.aperture.plot import (
    _ProjectedProfile,
    _cartesian_to_curvilinear,
    _curvilinear_to_cartesian,
    _pipe_projection_polygons,
    _profile_projection_segment,
    _shape_projection_extents,
)
from xtrack.aperture.structures import (
    Circle,
    Ellipse,
    Octagon,
    Polygon,
    Racetrack,
    Rectangle,
    RectEllipse,
)
from xtrack.aperture.transform import transform_matrix


@pytest.mark.parametrize(
    ('shape_class', 'shape_kwargs', 'direction', 'expected_support'),
    [
        (Circle, {'radius': 2.0}, [1.0, 0.0], 2.0),
        (Ellipse, {'half_major': 3.0, 'half_minor': 2.0}, [0.6, 0.8], np.hypot(1.8, 1.6)),
        (Rectangle, {'half_width': 3.0, 'half_height': 2.0}, [0.6, 0.8], 3.4),
        (
            Racetrack,
            {'half_width': 3.0, 'half_height': 2.0, 'half_major': 0.5, 'half_minor': 0.25},
            [0.6, 0.8],
            2.5 * 0.6 + 1.75 * 0.8 + np.hypot(0.5 * 0.6, 0.25 * 0.8),
        ),
    ],
)
def test_shape_projection_extents_for_symmetric_shapes(
    shape_class,
    shape_kwargs,
    direction,
    expected_support,
):
    shape = shape_class(**shape_kwargs)
    direction = np.asarray(direction)
    direction = direction / np.linalg.norm(direction)

    lower, upper = _shape_projection_extents(shape, direction)

    np.testing.assert_allclose([lower, upper], [-expected_support, expected_support], atol=1e-14, rtol=0)


def test_shape_projection_extents_for_octagon_and_polygon():
    direction = np.array([1.0, 1.0]) / np.sqrt(2.0)
    octagon = Octagon(half_width=2.0, half_height=1.5, half_diagonal=2.0)
    polygon = Polygon(vertices=np.array([
        [-2.0, -1.0],
        [3.0, -1.0],
        [1.0, 2.0],
        [-2.0, -1.0],
    ]))

    np.testing.assert_allclose(_shape_projection_extents(octagon, direction), [-2.0, 2.0], atol=1e-14)

    vertices = polygon.vertices.to_nparray()[:-1]
    expected = vertices @ direction
    np.testing.assert_allclose(
        _shape_projection_extents(polygon, direction),
        [np.min(expected), np.max(expected)],
        atol=1e-14,
    )


def test_rectellipse_projection_matches_dense_boundary_search():
    shape = RectEllipse(
        half_width=1.1,
        half_height=0.7,
        half_major=1.5,
        half_minor=1.0,
    )
    direction = np.array([0.37, 0.93])
    direction /= np.linalg.norm(direction)

    lower, upper = _shape_projection_extents(shape, direction)

    half_width = float(shape.half_width)
    half_height = float(shape.half_height)
    half_major = float(shape.half_major)
    half_minor = float(shape.half_minor)
    angles = np.linspace(0.0, 2.0 * np.pi, 100_000, endpoint=False)
    ellipse = np.column_stack([
        half_major * np.cos(angles),
        half_minor * np.sin(angles),
    ])
    inside_rectangle = (
        (np.abs(ellipse[:, 0]) <= half_width)
        & (np.abs(ellipse[:, 1]) <= half_height)
    )
    rectangle_x = np.linspace(-half_width, half_width, 10_000)
    rectangle_y = np.linspace(-half_height, half_height, 10_000)
    rectangle_boundary = np.vstack([
        np.column_stack([rectangle_x, np.full_like(rectangle_x, half_height)]),
        np.column_stack([rectangle_x, np.full_like(rectangle_x, -half_height)]),
        np.column_stack([np.full_like(rectangle_y, half_width), rectangle_y]),
        np.column_stack([np.full_like(rectangle_y, -half_width), rectangle_y]),
    ])
    inside_ellipse = (
        (rectangle_boundary[:, 0] / half_major) ** 2
        + (rectangle_boundary[:, 1] / half_minor) ** 2
        <= 1.0
    )
    boundary = np.vstack([ellipse[inside_rectangle], rectangle_boundary[inside_ellipse]])
    values = np.sum(boundary * direction, axis=1)

    np.testing.assert_allclose([lower, upper], [np.min(values), np.max(values)], atol=2e-5, rtol=0)


def test_profile_projection_segment_and_plane_degeneracies():
    rectangle = Rectangle(half_width=2.0, half_height=1.0)

    segment = _profile_projection_segment(rectangle, np.identity(4), normal_axis=1)
    np.testing.assert_allclose(segment[:, 1], 0.0, atol=1e-14)
    np.testing.assert_allclose(np.sort(segment[:, 0]), [-2.0, 2.0], atol=1e-14)

    coplanar = transform_matrix(rot_x_rad=np.pi / 2)
    with pytest.raises(ValueError, match='coplanar'):
        _profile_projection_segment(rectangle, coplanar, normal_axis=1)

    parallel = transform_matrix(shift_y=1.0, rot_x_rad=np.pi / 2)
    assert _profile_projection_segment(rectangle, parallel, normal_axis=1) is None


@pytest.mark.parametrize('curvature', [-0.2, 0.0, 0.2])
def test_curvilinear_cartesian_roundtrip(curvature):
    points = np.array([
        [-0.1, 0.2, 1.0],
        [0.3, -0.4, 4.0],
    ])

    cartesian = _curvilinear_to_cartesian(points, curvature)
    roundtrip = _cartesian_to_curvilinear(cartesian, curvature, reference_s=2.5)

    np.testing.assert_allclose(roundtrip, points, atol=1e-14, rtol=0)


def test_curved_pipe_projection_uses_maximum_angular_step():
    curvature = 0.5
    start = _ProjectedProfile(
        shift_s=0.0,
        endpoints_curvilinear=np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    end = _ProjectedProfile(
        shift_s=2.0,
        endpoints_curvilinear=np.array([[-0.5, 0.0, 2.0], [1.5, 0.0, 2.0]]),
    )

    polygons = _pipe_projection_polygons([start, end], curvature, max_curve_angle_rad=0.3)

    assert len(polygons) == 1
    # ceil(1 / 0.3) = 4 intervals, with five points on each side.
    assert polygons[0].shape == (10, 3)
    expected_endpoints = _curvilinear_to_cartesian(
        np.vstack([start.endpoints_curvilinear, end.endpoints_curvilinear]),
        curvature,
    )
    for endpoint in expected_endpoints:
        assert np.any(np.all(np.isclose(polygons[0], endpoint, atol=1e-14, rtol=0), axis=1))
