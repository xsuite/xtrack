import numpy as np
import pytest
import xobjects as xo
from scipy.special import ellipe

from xtrack.aperture import structures
from xtrack.aperture.kernels import build_aperture_kernels
from xtrack.aperture.structures import (
    Circle,
    Ellipse,
    Octagon,
    Profile,
    Racetrack,
    Rectangle,
    RectEllipse,
    ShapeTypes
)


@pytest.fixture(scope="module")
def context():
    context = xo.ContextCpu()
    build_aperture_kernels(context)
    return context


@pytest.fixture(scope="module")
def build_polygon_for_profile(context):
    def _build_polygon_for_profile(profile: ShapeTypes, num_points: int):
        profile = Profile(shape=profile)
        points = np.zeros((num_points, 2), dtype=np.float32)
        context.kernels.build_polygon_for_profile(points=points, num_points=num_points, profile=profile)
        return points

    return _build_polygon_for_profile


def ellipse_circumference(a: float, b: float) -> float:
    return 4 * a * ellipe(1 + b**2 / a**2)


def assert_polyline(pts: np.ndarray):
    atol = 1e-6 # microns
    assert pts.ndim == 2 and pts.shape[1] == 2
    assert np.isfinite(pts).all()
    assert np.allclose(pts[0], pts[-1], atol=atol), f"polyline not closed: {pts[0]} vs {pts[-1]} (atol={atol})"


def unique_loop(pts: np.ndarray) -> np.ndarray:
    """Return the unique points (drop the repeated last == first)."""
    return pts[:-1]


def segment_lengths(pts: np.ndarray) -> np.ndarray:
    """Lengths of the explicit segments of the closed polyline (no extra wrap)."""
    diffs = np.diff(pts, axis=0)
    return np.linalg.norm(diffs, axis=1)


def assert_uniform_sampling(pts: np.ndarray, expected_circumference: float = None):
    lens = segment_lengths(pts)
    num_points = lens.shape[0]

    if True or expected_circumference is None:
        circumference = (num_points - 1) * np.mean(lens)
    else:
        assert (num_points - 1) * np.mean(lens) < expected_circumference
        circumference = expected_circumference

    min_ds = circumference / (num_points - 1) / np.sqrt(2)
    atol = 1e-6 # microns
    assert (lens >= min_ds - atol).all()


def assert_centered(pts: np.ndarray):
    pts = unique_loop(pts)
    atol = 5e-5 # 50 microns
    c = pts.mean(axis=0)
    assert np.allclose(c, (0.0, 0.0), atol=atol), f"centroid {c} not near origin (atol={atol})"


def assert_circle_boundary(pts: np.ndarray, r: float):
    pts = unique_loop(pts)
    atol = 1e-6 # microns
    rad = np.linalg.norm(pts, axis=1)
    assert np.allclose(rad, r, atol=atol), f"circle boundary mismatch (atol={atol})"


def assert_ellipse_boundary(pts: np.ndarray, a: float, b: float):
    pts = unique_loop(pts)
    atol = 1e-6 # microns
    x, y = pts[:, 0], pts[:, 1]
    val = (x / a) ** 2 + (y / b) ** 2
    assert np.allclose(val, 1.0, atol=atol), f"ellipse implicit mismatch (atol={atol})"


def assert_rectangle_boundary(pts: np.ndarray, w: float, h: float):
    pts = unique_loop(pts)
    atol = 1e-6 # microns
    x, y = np.abs(pts[:, 0]), np.abs(pts[:, 1])
    on_vert = np.isclose(x, w, atol=atol) & (y <= h + atol)
    on_horiz = np.isclose(y, h, atol=atol) & (x <= w + atol)
    assert (on_vert | on_horiz).all(), "Some points are not on rectangle boundary"


def assert_racetrack_boundary(pts: np.ndarray, w: float, h: float, a: float, b: float):
    """
    Racetrack = Minkowski sum of rectangle (w - a, h - b) and ellipse (a,b).
      - vertical edges: |x| = w + a
      - horizontal edges: |y| = h + b
      - arc edges: (|x| - w) ** 2 / a ** 2 + (|y| - h) ** 2) / b**2 = 1 for |x| > w - a and |y| > h - b
    """
    pts = unique_loop(pts)
    atol = 1e-6 # microns

    x, y = np.abs(pts[:, 0]), np.abs(pts[:, 1])
    on_vert = np.isclose(x, w, atol=atol, rtol=0) & (y <= h - b + atol)
    on_horiz = np.isclose(y, h, atol=atol, rtol=0) & (x <= w - a + atol)
    on_ellipse = np.isclose(((x - w + a)**2) / (a**2) + ((y - h + b)**2) / b**2, 1, atol=1e-1, rtol=0)
    in_corner = (x > w - a) & (y > h - b)
    on_arc = on_ellipse & in_corner

    assert (on_vert | on_horiz | on_arc).all(), "Some points are not on racetrack boundary"


def assert_rectellipse_union_boundary(pts: np.ndarray, w: float, h: float, a: float, b: float):
    """
    RectEllipse is an intersection of a rectangle with an ellipse. Check that the points lie on the boundary, by
    verifying that they are inside the intersection of the two plus some tolerance, and that the lie outside
    the intersection minus some tolerance.
    """
    pts = unique_loop(pts)
    atol = 1e-6  # microns

    x, y = np.abs(pts[:, 0]), np.abs(pts[:, 1])
    in_rect_plus_eps = (x < w + atol) & (y < h + atol)
    in_ellipse_plus_eps = (x / a) ** 2 + (y / b) ** 2 < 1 + atol
    in_union_plus_eps = in_rect_plus_eps & in_ellipse_plus_eps

    outside_rect_minus_eps = (x > w - atol) | (y > h - atol)
    outside_ellipse_minus_eps = (x / a) ** 2 + (y / b) ** 2 > 1 - atol
    outside_union_minus_eps = outside_rect_minus_eps | outside_ellipse_minus_eps

    assert (in_union_plus_eps & outside_union_minus_eps).all()


def assert_octagon_boundary(pts: np.ndarray, w: float, h: float, d: float):
    """
    Points lie either on |x| = w, |y| = h, or |y| = -|x| + sqrt(2) * d
    """
    pts = unique_loop(pts)
    atol = 1e-6 # microns

    x, y = np.abs(pts[:, 0]), np.abs(pts[:, 1])
    on_vert = np.isclose(x, w, atol=atol, rtol=0)
    on_horiz = np.isclose(y, h, atol=atol, rtol=0)
    on_corner = np.isclose(y, -x + np.sqrt(2) * d, atol=atol, rtol=0)

    assert (on_vert | on_horiz | on_corner).all(), "Some points are not on octagon boundary"


@pytest.mark.parametrize("r,n", [
    (1.0, 360),
    (2.5, 2048),
])
def test_circle_correctness_and_uniformity(build_polygon_for_profile, r, n):
    pts = build_polygon_for_profile(Circle(radius=r), n)
    assert pts.shape == (n, 2)
    assert_polyline(pts)

    assert_centered(pts)
    assert_circle_boundary(pts, r)

    assert_uniform_sampling(pts, 2 * np.pi * r)


@pytest.mark.parametrize("w,h,n", [
    (2.0, 1.0, 1200),
    (1.0, 1.0, 2000),
])
def test_rectangle_correctness_and_uniformity(build_polygon_for_profile, w, h, n):
    pts = build_polygon_for_profile(Rectangle(half_width=w, half_height=h), n)
    assert pts.shape == (n, 2)
    assert_polyline(pts)

    assert_centered(pts)
    assert_rectangle_boundary(pts, w, h)
    assert_uniform_sampling(pts, 4 * (w + h))


@pytest.mark.parametrize("a,b,n", [
    (2.0, 1.0, 1200),
    (5.0, 0.5, 4000),
])
def test_ellipse_correctness_and_uniformity(build_polygon_for_profile, a, b, n):
    pts = build_polygon_for_profile(Ellipse(half_major=a, half_minor=b), n)
    assert pts.shape == (n, 2)
    assert_polyline(pts)

    assert_centered(pts)
    assert_ellipse_boundary(pts, a, b)
    assert_uniform_sampling(pts, ellipse_circumference(a, b))


@pytest.mark.parametrize("w,h,a,b,n", [
    (2.0, 1.0, 0.5, 0.5, 3500),
    (2.0, 1.0, 1.0, 0.25, 6000),
])
def test_racetrack_correctness_and_uniformity(build_polygon_for_profile, w, h, a, b, n):
    pts = build_polygon_for_profile(Racetrack(half_width=w, half_height=h, half_major=a, half_minor=b), n)
    assert pts.shape == (n, 2)
    assert_polyline(pts)

    assert_centered(pts)
    assert_racetrack_boundary(pts, w, h, a, b)
    assert_uniform_sampling(pts, 4 * (w + h) + ellipse_circumference(a, b))


def test_racetrack_degenerate_to_ellipse(build_polygon_for_profile):
    a, b = 2.0, 1.0
    w, h = a, b
    n = 2500

    pts = build_polygon_for_profile(Racetrack(half_width=w, half_height=h, half_major=a, half_minor=b), n)
    assert pts.shape == (n, 2)
    assert_polyline(pts)

    assert_ellipse_boundary(pts, a, b)
    assert_uniform_sampling(pts, ellipse_circumference(a, b))


@pytest.mark.parametrize("w,h,a,b,n", [
    (0.5, 0.25, 2.0, 1.0, 3500),  # rectangle inside ellipse
    (3.0, 2.0, 1.0, 0.5, 3500),   # ellipse inside rectangle
    (1.5, 0.6, 1.0, 1.2, 5000),   # barrel-y shape
    (2.0, 1.0, 2.3, 1.3, 300),    # 8-segment shape
    (0.022, 0.01715, 0.022, 0.022, 300), # LHC-style screen
])
def test_rectellipse_correctness_and_uniformity(build_polygon_for_profile, w, h, a, b, n):
    pts = build_polygon_for_profile(RectEllipse(half_width=w, half_height=h, half_major=a, half_minor=b), n)
    assert pts.shape == (n, 2)
    assert_polyline(pts)

    assert_centered(pts)
    assert_rectellipse_union_boundary(pts, w, h, a, b)
    assert_uniform_sampling(pts)


@pytest.mark.parametrize("w,h,d,n", [
    (2.0, 1.5, 1.8, 2500),
    (3.0, 2.0, 2.5, 4000),
    (2.0, 2.0, 1.5, 5000),
])
def test_octagon_correctness_and_uniformity(build_polygon_for_profile, w, h, d, n):
    pts = build_polygon_for_profile(Octagon(half_width=w, half_height=h, half_diagonal=d), n)
    assert pts.shape == (n, 2)
    assert_polyline(pts)

    assert_centered(pts)
    assert_octagon_boundary(pts, w, h, d)

    assert_uniform_sampling(pts, 4 * (w + h) + 4 * d * (np.sqrt(2) - 2))


@pytest.mark.parametrize("shape_class,params", [
    ('Circle', {'radius': 1.0}),
    ('Rectangle', {'half_width': 1.0, 'half_height': 0.5}),
    ('Ellipse', {'half_major': 1.5, 'half_minor': 1.0}),
    ('RectEllipse', {'half_width': 1.0, 'half_height': 0.5, 'half_major': 1.2, 'half_minor': 0.8}),
    ('Racetrack', {'half_width': 1.0, 'half_height': 0.5, 'half_major': 0.3, 'half_minor': 0.2}),
    ('Octagon', {'half_width': 1.0, 'half_height': 0.8, 'half_diagonal': 0.8}),
])
def test_smoke_small_num_points(build_polygon_for_profile, shape_class, params):
    profile = getattr(structures, shape_class)(**params)
    n = 8
    pts = build_polygon_for_profile(profile, n)
    assert pts.shape == (n, 2)
    assert_polyline(pts)
    lens = segment_lengths(pts)
    assert lens.min() > 0
