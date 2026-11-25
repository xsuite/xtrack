"""Check SVG profiles at the polygon API and aperture-model boundaries."""

import numpy as np
import pytest

import xtrack as xt
from xtrack.aperture.plot import pipe_projection, pipe_solid
from xtrack.aperture.structures import Profile, SVGShape


RECTANGLE_PATH = 'M-2,-1 L2,-1 L2,1 L-2,1 Z'
CIRCLE_PATH = 'M10,0 A10,10 0 0,1 -10,0 A10,10 0 0,1 10,0 Z'


def _make_line(svg_path=RECTANGLE_PATH, resolution=1e-4):
    """Create a one-metre pipe with identical SVG profiles at both ends.

    Return the unbuilt line; its environment owns 'profile' and 'pipe'.
    """
    env = xt.Environment()
    drift = env.new('drift', xt.Drift, length=1)
    line = env.new_line(name='line', components=[drift])

    profile = env.aperture.new_profile(
        'profile', 'SVGShape', svg_path=svg_path, resolution=resolution,
    )
    pipe = env.aperture.new_pipe(
        'pipe', positions=[profile.place(), profile.place(at=1)],
    )
    line.aperture.place_pipe('installed', pipe, at='drift')

    # 48 perimeter intervals put samples on all rectangular corners.
    line.aperture.num_profile_points = 49
    line.aperture._with_progress = False
    line.aperture._is_ring_setting = False

    return line


def test_svg_defaults_and_validation_at_build_time():
    """Field defaults apply at construction; invalid geometry fails on build."""
    shape = SVGShape(svg_path='M0,0 L1,0 L1,1')

    assert shape.scale == 0.001
    assert shape.resolution == 1e-4

    with pytest.raises(ValueError, match='closed'):
        shape.build_polygon(9)


def test_nonuniform_bezier_is_resampled_uniformly():
    """A straight cubic edge with x(t)=t**3 must be uniformly resampled.

    The known square perimeter also checks closure, y-axis inversion, and
    anticlockwise orientation, independently of the intermediate polygon.
    """
    shape = SVGShape(
        svg_path='M0,0 C0,0 0,0 1,0 L1,-1 L0,-1 Z', scale=1, resolution=0.2,
    )

    points = shape.build_polygon(9)

    np.testing.assert_allclose(points, [
        [0, 0], [0.5, 0], [1, 0], [1, 0.5],
        [1, 1], [0.5, 1], [0, 1], [0, 0.5], [0, 0],
    ], atol=1e-14)


def test_profile_writes_only_into_the_requested_output_view():
    """Profile delegation must fill the supplied row without touching neighbours.

    Model building passes such views into shared polygon storage, so check
    both object identity and equivalence to the allocating API.
    """
    profile = Profile(shape=SVGShape(svg_path=RECTANGLE_PATH))
    storage = np.full((3, 25, 2), np.nan)
    output_view = storage[1]

    returned_points = profile.build_polygon(25, out_poly=output_view)

    assert returned_points is output_view
    assert np.isnan(storage[[0, 2]]).all()

    np.testing.assert_array_equal(output_view, profile.build_polygon(25))


def test_model_rebuild_applies_updated_sampling_settings():
    """Rebuilding must apply changes to point count and SVG resolution.

    Change the output count from 49 to 97 and check the returned array size.
    Then keep 97 points, refine the SVG resolution from 10 mm to 0.1 mm,
    and check that the sampled radii match the known 10 mm circle to 1 micron.
    """
    line = _make_line(CIRCLE_PATH, resolution=0.01)
    profile = line.env.aperture.profiles['profile']

    line.aperture.build_model()
    coarse_points = line.aperture.get_cross_sections_at_s([0.5]).cross_section[0]

    line.aperture.num_profile_points = 97
    line.aperture.build_model()
    more_points = line.aperture.get_cross_sections_at_s([0.5]).cross_section[0]

    profile.shape.resolution = 1e-4
    line.aperture.build_model()
    refined_points = line.aperture.get_cross_sections_at_s([0.5]).cross_section[0]

    assert coarse_points.shape == (49, 2)
    assert more_points.shape == refined_points.shape == (97, 2)

    np.testing.assert_allclose(
        np.linalg.norm(refined_points, axis=1), 0.01, atol=1e-6, rtol=0,
    )


@pytest.mark.parametrize('built', [False, True], ids=['unbuilt', 'built'])
def test_svg_parameters_survive_json_and_model_lifecycle(tmp_path, built):
    """Preserve the SVG path, scale, and resolution through packing and JSON.

    Test both standalone and packed definitions. Reloading and discarding a
    built model exercise the positional dictionary and cross-buffer copies
    required by Xobjects unionrefs; rebuilding must recover the same geometry.
    """
    line = _make_line()
    env = line.env
    profile = env.aperture.profiles['profile']
    profile.shape.scale = 0.002
    profile.shape.resolution = 2e-4

    expected_fields = {
        'svg_path': RECTANGLE_PATH, 'scale': 0.002, 'resolution': 2e-4,
    }
    if built:
        line.aperture.build_model()

    json_path = tmp_path / 'svg.json'
    env.aperture.to_json(json_path)

    restored_env = xt.Environment.from_dict(env.to_dict(), with_progress=False)
    restored_env.aperture.from_json(json_path)
    restored_profile = restored_env.aperture.profiles['profile']

    assert isinstance(restored_profile.shape, SVGShape)
    assert restored_profile.shape._to_dict() == expected_fields

    restored_line = restored_env.lines['line']
    restored_line.aperture._with_progress = False
    restored_line.aperture.build_model()
    restored_points = restored_line.aperture.get_cross_sections_at_s([0.5]).cross_section[0]

    np.testing.assert_allclose(
        [restored_points.min(axis=0), restored_points.max(axis=0)],
        [[-0.004, -0.002], [0.004, 0.002]], atol=1e-14,
    )

    env.aperture.discard_model()
    assert profile.shape._to_dict() == expected_fields

    line.aperture.build_model()
    rebuilt_points = line.aperture.get_cross_sections_at_s([0.5]).cross_section[0]

    np.testing.assert_allclose(rebuilt_points, restored_points, atol=1e-14)


def test_from_limits_preserves_svg_parameters_and_scaled_geometry():
    """Import an SVG LimitPolygon as a parametric SVGShape with correct units."""
    limit = xt.LimitPolygon(svg={'path': RECTANGLE_PATH, 'scale': 0.002})
    line = xt.Line(elements=[limit, xt.Drift(length=1)])
    line.aperture.num_profile_points = 49

    line.aperture.from_limits(with_progress=False)

    profile = line.env.aperture.profiles[line.element_names[0]]
    assert isinstance(profile.shape, SVGShape)
    assert str(profile.shape.svg_path) == RECTANGLE_PATH
    assert profile.shape.scale == 0.002

    section = line.aperture.get_cross_sections_at_s([0], extents=True)

    np.testing.assert_allclose([section.min_x[0], section.max_x[0]], [-0.004, 0.004])
    np.testing.assert_allclose([section.min_y[0], section.max_y[0]], [-0.002, 0.002])


def test_svg_plot_geometry_has_expected_pipe_bounds():
    """Projection polygons and 3D rings must span the same rectangular pipe.

    Exercise the geometry used by floor and 3D plots without rendering or
    depending on matplotlib. Compare actual bounds rather than only finiteness.
    """
    line = _make_line()
    pipe = line.env.aperture.pipes['pipe']

    projection = pipe_projection(pipe, np.eye(4))
    projection_points = np.concatenate(projection.polygons)

    np.testing.assert_allclose(
        [projection_points.min(axis=0), projection_points.max(axis=0)],
        [[0, -0.002], [1, 0.002]], atol=1e-14,
    )

    solid = pipe_solid(pipe, len_points=49)
    ring_points = np.concatenate(solid.profile_rings)

    np.testing.assert_allclose(
        [ring_points.min(axis=0), ring_points.max(axis=0)],
        [[-0.002, -0.001, 0], [0.002, 0.001, 1]], atol=1e-14,
    )
