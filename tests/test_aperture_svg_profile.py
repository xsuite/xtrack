import sys

import numpy as np
import pytest

import xtrack as xt
from xtrack.aperture.plot import (
    _shape_projection_extents, pipe_projection, pipe_solid,
)
from xtrack.aperture.profile_converters import profile_from_limit_element
from xtrack.aperture.structures import Circle, Polygon, Profile, SVGShape
from xtrack.svgutils import svg_to_points_by_resolution


RECT = 'M-2,-1 L2,-1 L2,1 L-2,1 Z'
CIRCLE = 'M10,0 A10,10 0 0,1 -10,0 A10,10 0 0,1 10,0 Z'


def signed_area(points):
    return np.sum(points[:-1, 0] * points[1:, 1]
                  - points[1:, 0] * points[:-1, 1]) / 2


def make_model(path=RECT, **shape_kwargs):
    env = xt.Environment()
    line = env.new_line(name='line', components=[env.new('d', xt.Drift, length=1)])
    profile = env.aperture.new_profile('svg', 'SVGShape', svg_data=path, **shape_kwargs)
    pipe = env.aperture.new_pipe('pipe', positions=[profile.place(), profile.place(at=1)])
    line.aperture.place_pipe('installed', pipe, at='d')
    line.aperture.num_profile_points = 49
    line.aperture._with_progress = False
    line.aperture._is_ring_setting = False
    return env, line, profile, pipe


@pytest.mark.parametrize('path', [RECT, 'M-2,-1 L-2,1 L2,1 L2,-1 Z', CIRCLE,
                                      'M0,0 C0,3 2,-1 2,0 Z'])
def test_svg_uses_existing_polygon_resampler(path):
    shape = SVGShape(svg_data=path, resolution=1e-5)
    vertices = svg_to_points_by_resolution(path, resolution=shape.resolution, scale=shape.scale)
    expected = Polygon(vertices=vertices).build_polygon(129)
    actual = shape.build_polygon(129)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual[-1], actual[0])
    assert signed_area(actual) > 0


def test_nonuniform_straight_bezier_is_resampled_uniformly():
    # First edge has x(t)=t**3. Compare to the same outline drawn with lines.
    curved = SVGShape(svg_data='M0,0 C0,0 0,0 1,0 L1,1 L0,1 Z', scale=1, resolution=0.2)
    straight = SVGShape(svg_data='M0,0 L1,0 L1,1 L0,1 Z', scale=1)
    np.testing.assert_allclose(curved.build_polygon(17), straight.build_polygon(17), atol=1e-14)


def test_output_buffer_and_profile_delegation():
    shape = SVGShape(svg_data=RECT)
    assert shape.scale == 0.001
    assert shape.resolution == 1e-4
    assert shape.valid()
    out = np.full((3, 25, 2), np.nan)
    profile = Profile(shape=shape)
    assert profile.build_polygon(25, out_poly=out[1]) is not None
    np.testing.assert_array_equal(out[1], shape.build_polygon(25))
    assert np.isnan(out[[0, 2]]).all()
    target = out[0]
    assert shape.build_polygon(25, out_poly=target) is target
    circle = Circle(radius=0.01)
    target = out[2]
    assert circle.build_polygon(25, out_poly=target) is target
    np.testing.assert_allclose(np.linalg.norm(target, axis=1), 0.01)


@pytest.mark.parametrize('len_points', [-1, 0, 1, 2])
def test_invalid_point_count(len_points):
    for shape in [SVGShape(svg_data=RECT), Circle(radius=1)]:
        with pytest.raises(ValueError, match='len_points'):
            shape.build_polygon(len_points)


@pytest.mark.parametrize('out', [np.empty((4, 2)), np.empty((5, 2), dtype=np.float32),
                                 np.empty((5, 4))[:, ::2]])
def test_invalid_output_buffer(out):
    with pytest.raises(ValueError, match='out_poly'):
        SVGShape(svg_data=RECT).build_polygon(5, out_poly=out)


def test_constructor_validation_and_dictionary_rehydration():
    with pytest.raises(ValueError, match='closed'):
        SVGShape(svg_data='M0,0 L1,0 L1,1')
    with pytest.raises(ValueError, match='resolution'):
        SVGShape(svg_data=RECT, resolution=0)
    shape = SVGShape({'svg_data': RECT, 'scale': 0.002, 'resolution': 2e-6})
    assert shape.scale == 0.002
    assert shape.resolution == 2e-6
    shape.resolution = -1
    assert not shape.valid()


def test_resolution_and_final_point_count_are_independent():
    shape = SVGShape(svg_data=CIRCLE, resolution=0.01)
    errors = []
    for resolution in [0.01, 0.002, 0.0002]:
        shape.resolution = resolution
        points = shape.build_polygon(513)
        errors.append(np.max(0.01 - np.linalg.norm(points, axis=1)))
    assert errors[2] < errors[1] < errors[0]
    shape.resolution = 0.01
    # More final points still leave the same intermediate polygon error.
    fine = shape.build_polygon(1025)
    assert np.max(0.01 - np.linalg.norm(fine, axis=1)) > 1e-4


@pytest.mark.parametrize('path,convex', [(RECT, True),
    ('M-2,-1 L0,-1 L0,0 L2,0 L2,1 L-2,1 Z', False)])
def test_svg_model_cross_sections_and_convexity(path, convex):
    env, line, profile, pipe = make_model(path)
    line.aperture.build_model()
    sections = line.aperture.get_cross_sections_at_s([0.25, 0.5, 0.75], extents=True)
    assert np.isfinite(sections.cross_section).all()
    np.testing.assert_allclose(sections.cross_section[0], sections.cross_section[1], atol=1e-14)
    # Exercise the existing C convexity computation for interpolated sections.
    aperture = line.aperture
    is_convex = np.zeros(3, dtype=np.int8)
    aperture._model.cross_sections_at_s(
        survey_at_s=aperture._survey_data.resample(np.array([0.25, 0.5, 0.75])),
        profile_polygons=aperture._profile_polygons,
        aperture_bounds=aperture._aperture_bounds,
        survey=aperture._survey_data,
        cross_sections=sections.cross_section,
        tol_r=sections.tol_r, tol_x=sections.tol_x, tol_y=sections.tol_y,
        is_convex=is_convex,
    )
    assert np.all(is_convex == convex)
    if convex:
        np.testing.assert_allclose(sections.min_x, -0.002, atol=1e-14)
        np.testing.assert_allclose(sections.max_x, 0.002, atol=1e-14)
        np.testing.assert_allclose(sections.min_y, -0.001, atol=1e-14)
        np.testing.assert_allclose(sections.max_y, 0.001, atol=1e-14)


@pytest.mark.parametrize('built', [False, True])
@pytest.mark.parametrize('format', ['dict', 'json'])
def test_svg_serialization_and_discard(tmp_path, built, format):
    env, line, profile, pipe = make_model(scale=0.002, resolution=2e-6)
    if built:
        line.aperture.build_model()
    expected = profile.shape._to_dict()
    assert expected == {'svg_data': RECT, 'scale': 0.002, 'resolution': 2e-6}
    restored = xt.Environment.from_dict(env.to_dict(), with_progress=False)
    if format == 'dict':
        restored.aperture.from_dict(env.aperture.to_dict())
    else:
        path = tmp_path / 'svg.json'
        env.aperture.to_json(path)
        restored.aperture.from_json(path)
    assert restored.aperture.profiles['svg'].shape._to_dict() == expected
    restored.lines['line'].aperture._with_progress = False
    restored.lines['line'].aperture.build_model()
    np.testing.assert_allclose(
        restored.lines['line'].aperture.get_cross_sections_at_s([0.5], extents=True).min_x,
        -0.004, atol=1e-14)
    assert restored.aperture.profiles['svg'].shape._to_dict() == expected
    env.aperture.discard_model()
    assert profile.shape._to_dict() == expected
    line.aperture.build_model()
    np.testing.assert_allclose(
        line.aperture.get_cross_sections_at_s([0.5], extents=True).min_x, -0.004, atol=1e-14)


def test_svg_limit_conversion_preserves_path():
    limit = xt.LimitPolygon(svg={'path': RECT, 'scale': 0.002, 'curved_steps': 25})
    shape, dx, dy = profile_from_limit_element(limit)
    assert isinstance(shape, SVGShape)
    assert str(shape.svg_data) == RECT
    assert shape.scale == 0.002
    assert (dx, dy) == (0, 0)
    plain, _, _ = profile_from_limit_element(
        xt.LimitPolygon(x_vertices=[-1, 1, 1, -1], y_vertices=[-1, -1, 1, 1]))
    assert isinstance(plain, Polygon)
    line = xt.Line(elements=[limit, xt.Drift(length=1)])
    line.aperture.num_profile_points = 49
    line.aperture._with_progress = False
    line.aperture.from_limits(with_progress=False)
    assert isinstance(line.env.aperture.profiles[line.element_names[0]].shape, SVGShape)
    line.aperture.build_model()
    np.testing.assert_allclose(line.aperture.get_cross_sections_at_s([0], extents=True).min_x, -0.004, atol=1e-14)


def test_svg_plot_geometry_without_matplotlib():
    was_loaded = 'matplotlib.pyplot' in sys.modules
    env, line, profile, pipe = make_model()
    np.testing.assert_allclose(_shape_projection_extents(profile.shape, np.array([1., 0.])),
                               [-0.002, 0.002])
    np.testing.assert_allclose(_shape_projection_extents(profile.shape, np.array([0., 1.])),
                               [-0.001, 0.001])
    solid = pipe_solid(pipe, len_points=49)
    assert len(solid.profile_rings) == 2
    assert all(np.isfinite(face).all() for face in solid.faces)
    projection = pipe_projection(pipe, np.eye(4))
    assert projection.polygons
    assert all(np.isfinite(polygon).all() for polygon in projection.polygons)
    assert ('matplotlib.pyplot' in sys.modules) == was_loaded


def test_model_rebuild_uses_current_resolution_and_point_count():
    env, line, profile, pipe = make_model(CIRCLE, resolution=0.01)
    line.aperture.build_model()
    coarse = line.aperture.get_cross_sections_at_s([0.5]).cross_section[0]
    profile.shape.resolution = 0.0001
    line.aperture.num_profile_points = 97
    line.aperture.build_model()
    fine = line.aperture.get_cross_sections_at_s([0.5]).cross_section[0]
    assert coarse.shape == (49, 2)
    assert fine.shape == (97, 2)
    assert np.max(0.01 - np.linalg.norm(fine, axis=1)) < np.max(
        0.01 - np.linalg.norm(coarse, axis=1))
