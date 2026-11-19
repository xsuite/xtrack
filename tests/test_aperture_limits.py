import pickle

import numpy as np
import pytest

import xtrack as xt


def make_line(shape='Rectangle', parameters=None, *, curve=0., roll=0., thin=False, **position):
    env = xt.Environment()
    env['gap'] = .03
    env['offset'] = .002
    env['roll'] = .2
    env['pitch'] = 0.
    env['end'] = 2.
    profile = env.aperture.new_profile('profile', shape,
        **(parameters if parameters is not None else dict(max_x='gap', max_y=.1)))
    element = env.new('element', 'Bend', length=2., angle=curve * 2, rot_s_rad=roll)
    line = env.new_line(name='line', components=[element])
    pipe = env.aperture.new_pipe('pipe', curvature=curve, positions=[
        profile.place(at=s, **position) for s in ([0.] if thin else [0., 2.])])
    line.aperture.place_pipe('installed', pipe, at='element', rot_z_rad=roll)
    return env, line


@pytest.mark.parametrize('shape, parameters, expected', [
    ('Circle', dict(radius=.03), xt.LimitEllipse),
    ('Rectangle', dict(max_x=.03, max_y=.02), xt.LimitRect),
    ('Ellipse', dict(a=.03, b=.02), xt.LimitEllipse),
    ('RectEllipse', dict(max_x=.03, max_y=.02, a=.04, b=.025), xt.LimitRectEllipse),
    ('Racetrack', dict(max_x=.03, max_y=.02, a=.01, b=.005), xt.LimitRacetrack),
    ('Octagon', dict(max_x=.03, max_y=.02, max_diagonal=.03), xt.LimitPolygon),
    ('Polygon', dict(vertices=[[-.03, -.02], [.03, -.02], [.025, .02], [-.03, .02]]), xt.LimitPolygon),
])
def test_native_limits_without_resampling(shape, parameters, expected, monkeypatch):
    env, line = make_line(shape, parameters)
    line.aperture.num_profile_points = 9
    def no_sampling(*args, **kwargs):
        raise AssertionError('Exact sections must not resample their cross-sections.')
    monkeypatch.setattr(line.aperture, 'get_cross_sections_at_s', no_sampling)
    limit = line.aperture.get_limit_elements_at_s([1.])[1.]
    assert isinstance(limit, expected)
    if shape == 'Circle':
        assert limit.a == limit.b == .03
    elif shape == 'Polygon':
        np.testing.assert_array_equal(limit.x_vertices, np.array(parameters['vertices'])[:, 0])
        assert len(limit.x_vertices) == 4
    elif shape == 'Octagon':
        assert len(limit.x_vertices) == 8
    else:
        for field, value in parameters.items():
            assert getattr(limit, field) == value


@pytest.mark.parametrize('curve, roll', [(0., 0.), (.1, 0.), (-.1, .4), (.1, np.pi / 2)])
def test_native_curved_pipe_and_transverse_expressions(curve, roll):
    env, line = make_line(curve=curve, roll=roll, shift_x='offset', rot_s_rad='roll')
    limits = line.aperture.get_limit_elements_at_s([.3, 1., 1.7], names=['a', 'b', 'c'])
    for limit in limits.values():
        assert isinstance(limit, xt.LimitRect)
        assert limit.max_x == .03
        np.testing.assert_allclose([limit.shift_x, limit.shift_y],
            [.002 * np.cos(roll), .002 * np.sin(roll)], atol=1e-12)
        np.testing.assert_allclose(limit.rot_s_rad, roll + .2, atol=1e-12)
    env['gap'] = .05
    env['offset'] = .006
    env['roll'] = -.1
    for limit in limits.values():
        assert limit.max_x == .05 and limit.min_x == -.05
        np.testing.assert_allclose([limit.shift_x, limit.shift_y],
            [.006 * np.cos(roll), .006 * np.sin(roll)], atol=1e-12)
        np.testing.assert_allclose(limit.rot_s_rad, roll - .1, atol=1e-12)


@pytest.mark.parametrize('shape, parameters, expected', [
    ('Circle', dict(radius='gap'), xt.LimitEllipse),
    ('Rectangle', dict(max_x='gap', max_y=.1), xt.LimitRect),
    ('Ellipse', dict(a='gap', b='2*gap'), xt.LimitEllipse),
    ('RectEllipse', dict(max_x='gap', max_y=.1, a='2*gap', b=.12), xt.LimitRectEllipse),
    ('Racetrack', dict(max_x='gap', max_y=.1, a='gap/4', b=.02), xt.LimitRacetrack),
])
def test_native_shape_expressions_copy_and_json(shape, parameters, expected, tmp_path):
    env, line = make_line(shape, parameters)
    limit = line.aperture.get_limit_elements_at_s([1.], names=['limit'])[1.]
    assert isinstance(limit, expected)
    line.insert(env.place('limit', at=1.))
    env.to_json(tmp_path / 'model.json', include_aperture=True)
    restored = xt.Environment.from_json(tmp_path / 'model.json', with_progress=False)
    for target in (env, line.copy().env, env.copy(with_progress=False),
                   restored, pickle.loads(pickle.dumps(env))):
        target['gap'] = .04
        current = target['limit']
        for field, expression in parameters.items():
            expected_value = target.eval(expression) if isinstance(expression, str) else expression
            if field == 'radius':
                np.testing.assert_allclose(current.b, expected_value, atol=1e-14)
                field = 'a'
            np.testing.assert_allclose(getattr(current, field), expected_value, atol=1e-14)
        if hasattr(current, 'a_b_squ'):
            np.testing.assert_allclose(current.a_b_squ, current.a**2 * current.b**2, rtol=1e-14)


def test_standalone_expressions_warn_and_are_snapshots():
    env, line = make_line()
    with pytest.warns(UserWarning, match=r'not translated.*profile.*max_x'):
        limit = line.aperture.get_limit_elements_at_s([.5])[.5]
    env['gap'] = .06
    assert limit.max_x == .03
    with pytest.warns(UserWarning, match='standalone'):
        polygon = line.aperture.get_limit_elements_at_s([.5], prefer_native=False)[.5]
    assert isinstance(polygon, xt.LimitPolygon)


def test_polygon_fallback_warns_for_oblique_and_twisted_sections():
    env, line = make_line(rot_x_rad=.1)
    with pytest.warns(UserWarning, match=r'not translated.*max_x'):
        limit = line.aperture.get_limit_elements_at_s([1.], names=['oblique'])[1.]
    assert isinstance(limit, xt.LimitPolygon)
    frozen = limit.x_vertices.copy()
    env['gap'] = .04
    np.testing.assert_array_equal(limit.x_vertices, frozen)

    env, line = make_line(rot_s_rad=0.)
    env.aperture.pipes['pipe'][1].rot_s_rad = .3
    with pytest.warns(UserWarning, match='not translated'):
        limit = line.aperture.get_limit_elements_at_s([1.], names=['twisted'])[1.]
    assert isinstance(limit, xt.LimitPolygon)


def test_untranslated_geometry_and_tolerance_expressions_warn():
    env, line = make_line(rot_x_rad='pitch', tol_x='offset')
    with pytest.warns(UserWarning, match='rot_x_rad') as messages:
        limit = line.aperture.get_limit_elements_at_s([1.], names=['limit'])[1.]
    assert 'tol_x' in str(messages[0].message)
    assert 'max_x' not in str(messages[0].message)
    assert isinstance(limit, xt.LimitRect)
    env['gap'] = .05
    assert limit.max_x == .05


def test_interpolated_circles_retain_radius_and_offset_expressions():
    env, line = make_line('Circle', dict(radius='gap'))
    second = env.aperture.new_profile('exit', 'Circle', radius='2*gap')
    position = env.aperture.pipes['pipe'][1]
    position.profile_index = env.aperture.profile_index(second)
    position.shift_x = 'offset'
    position.shift_s = 'end'
    limit = line.aperture.get_limit_elements_at_s([1.], names=['limit'])[1.]
    assert isinstance(limit, xt.LimitEllipse)
    np.testing.assert_allclose([limit.a, limit.shift_x], [.045, .001], atol=1e-14)
    env['end'] = 1.5
    env['gap'] = .06
    env['offset'] = .03
    np.testing.assert_allclose([limit.a, limit.b, limit.shift_x], [.1, .1, .02], atol=1e-14)


def test_different_shapes_do_not_get_promoted_to_native_limits():
    env, line = make_line('Circle', dict(radius=.03))
    other = env.aperture.new_profile('rect', 'Rectangle', max_x=.02, max_y=.04)
    env.aperture.pipes['pipe'][1].profile_index = env.aperture.profile_index(other)
    limit = line.aperture.get_limit_elements_at_s([1.])[1.]
    assert isinstance(limit, xt.LimitPolygon)


def test_polygon_vertex_expressions_and_normals():
    env, line = make_line('Polygon', dict(vertices=[[-.03, -.02], [.03, -.02], [.03, .02], [-.03, .02]]))
    shape = env.aperture.profiles['profile'].shape
    shape.vertices[1, 0] = 'gap'
    shape.vertices[2, 0] = 'gap'
    limit = line.aperture.get_limit_elements_at_s([1.], names=['limit'])[1.]
    env['gap'] = .05
    np.testing.assert_allclose(limit.x_vertices, [-.03, .05, .05, -.03], atol=1e-14)
    fresh = xt.LimitPolygon(x_vertices=limit.x_vertices, y_vertices=limit.y_vertices)
    np.testing.assert_allclose(limit.x_normal, fresh.x_normal, atol=1e-14)
    np.testing.assert_allclose(limit.y_normal, fresh.y_normal, atol=1e-14)
    particles = xt.Particles(p0c=1e9, x=[.04, .06], y=[0., 0.])
    limit.track(particles)
    np.testing.assert_array_equal(particles.state[np.argsort(particles.particle_id)] > 0, [True, False])


def test_named_limit_validation_is_non_destructive():
    env, line = make_line()
    original = set(env.elements.keys())
    for names, positions, error in [(['a'], [1., 1.5], ValueError),
                                    (['a', 'a'], [1., 1.5], ValueError),
                                    (['element'], [1.], ValueError),
                                    ('name', [1.], TypeError)]:
        with pytest.raises(error):
            line.aperture.get_limit_elements_at_s(positions, names=names)
        assert set(env.elements.keys()) == original


def test_pipe_selection_uses_model_indices_not_sorted_table_indices():
    env = xt.Environment()
    env['gap'] = .03
    env.new('first', 'Drift', length=1.)
    env.new('second', 'Drift', length=1.)
    line = env.new_line(components=['first', 'second'])
    env.aperture.new_profile('circle', 'Circle', radius=.02)
    env.aperture.new_profile('rect', 'Rectangle', max_x='gap', max_y=.1)
    line.aperture.new_element_pipe('second', 'rect')
    line.aperture.new_element_pipe('first', 'circle')
    limits = line.aperture.get_limit_elements_at_s([1.5, .5], names=['r', 'c'])
    assert isinstance(limits[.5], xt.LimitEllipse)
    assert isinstance(limits[1.5], xt.LimitRect)
    env['gap'] = .04
    assert limits[1.5].max_x == .04


@pytest.mark.parametrize('curve, expected', [(0., xt.LimitEllipse), (.1, xt.LimitPolygon)])
def test_single_profile_is_projected_from_its_own_plane(curve, expected):
    env, line = make_line('Circle', dict(radius=.03), curve=curve, thin=True)
    limit = line.aperture.get_limit_elements_at_s([1.])[1.]
    assert isinstance(limit, expected)
    if curve:
        section = line.aperture.get_cross_sections_at_s([1.]).cross_section[0, :-1]
        np.testing.assert_array_equal(limit.x_vertices, section[:, 0])
        np.testing.assert_array_equal(limit.y_vertices, section[:, 1])


@pytest.mark.parametrize('shape', ['Polygon', 'Octagon'])
def test_exact_polygon_expressions_serialize_without_aperture_model(shape, tmp_path):
    if shape == 'Polygon':
        env, line = make_line(shape, dict(vertices=[
            [-.03, -.02], [.03, -.02], [.03, .02], [-.03, .02]]))
        env.aperture.profiles['profile'].shape.vertices[1, 0] = 'gap'
        env.aperture.profiles['profile'].shape.vertices[2, 0] = 'gap'
    else:
        env, line = make_line(shape, dict(max_x='gap', max_y=.02, max_diagonal=.03))
    line.aperture.get_limit_elements_at_s([1.], names=['limit'])
    line.insert(env.place('limit', at=1.))
    line.to_json(tmp_path / 'tracking.json')
    restored = xt.Line.from_json(tmp_path / 'tracking.json', with_progress=False)
    for current in (line.copy(), restored):
        current['gap'] = .035
        limit = current.element_dict['limit']
        np.testing.assert_allclose(max(limit.x_vertices), .035, atol=1e-14)
        assert len(limit.x_vertices) == (4 if shape == 'Polygon' else 8)
        fresh = xt.LimitPolygon(x_vertices=limit.x_vertices, y_vertices=limit.y_vertices)
        np.testing.assert_allclose(limit.x_normal, fresh.x_normal, atol=1e-14)
        np.testing.assert_allclose(limit.y_normal, fresh.y_normal, atol=1e-14)


def test_racetrack_outside_native_parameter_range_falls_back():
    env, line = make_line('Racetrack', dict(max_x=.03, max_y=.02, a=-.01, b=.005))
    limit = line.aperture.get_limit_elements_at_s([1.])[1.]
    assert isinstance(limit, xt.LimitPolygon)
    assert np.all(np.isfinite(limit.x_vertices))


@pytest.mark.parametrize('reversed_pipe', [False, True])
def test_native_taper_uses_pipe_distance_across_bend_exit(reversed_pipe):
    # The upstream profile plane meets a curved survey, the downstream plane
    # meets a straight one. Their survey-s separation differs from the pipe length.
    env = xt.Environment()
    line = env.new_line(components=[
        env.new('bend', 'Bend', length=1., angle=.3),
        env.new('straight', 'Drift', length=1.),
    ])
    small = env.aperture.new_profile('small', 'Circle', radius=.02)
    large = env.aperture.new_profile('large', 'Circle', radius=.04)
    pipe = env.aperture.new_pipe('taper', positions=[
        small.place(at=-.5), large.place(at=.5)])
    line.aperture.place_pipe('installed', pipe, at='straight',
                             rot_y_rad=np.pi if reversed_pipe else 0.)

    # Both planes must be found, including the one in the preceding segment.
    bounds = line.aperture.get_bounds_table()
    np.testing.assert_allclose(bounds.s, [1 - np.arcsin(.5 * .3) / .3, 1.5], atol=1e-12)

    # The plane 0.25 m after the bend is 3/4 (or 1/4) along the taper.
    limit = line.aperture.get_limit_elements_at_s([1.25])[1.25]
    assert isinstance(limit, xt.LimitEllipse)
    radius = .025 if reversed_pipe else .035
    np.testing.assert_allclose([limit.a, limit.b], radius, rtol=0, atol=1e-14)


@pytest.mark.parametrize('shape, tilt, roll, expected', [
    ('Circle', .2, .4, xt.LimitEllipse),
    ('RectEllipse', .002, .4, xt.LimitRectEllipse),
    ('RectEllipse', .2, .4, xt.LimitPolygon),
])
def test_oblique_uniform_bore(shape, tilt, roll, expected):
    env = xt.Environment()
    line = env.new_line(components=[env.new('drift', 'Drift', length=2.)])
    params = (dict(radius=.03) if shape == 'Circle' else
              dict(max_x=.025, max_y=.02, a=.03, b=.023))
    profile = env.aperture.new_profile('profile', shape, **params)
    pipe = env.aperture.new_pipe('pipe', positions=[
        profile.place(at=z, rot_s_rad=roll) for z in [0., 2.]])
    line.aperture.place_pipe('installed', pipe, at='drift', rot_y_rad=tilt)
    section = line.aperture.get_limit_elements_at_s([1.])[1.]
    assert isinstance(section, expected)
    if expected == xt.LimitPolygon:
        return
    np.testing.assert_allclose([section.shift_x, section.shift_y],
                               [np.tan(tilt), 0], atol=1e-10)
    if shape == 'Circle':
        np.testing.assert_allclose([section.a, section.b], [.03/np.cos(tilt), .03])
        np.testing.assert_allclose(np.sin(section.rot_s_rad), 0, atol=1e-12)
    else:
        np.testing.assert_allclose([section.max_x, section.max_y, section.a, section.b],
                                   [.025, .02, .03, .023], atol=1e-7)


def test_nearby_pipe_transitions_keep_the_sampled_contour():
    env = xt.Environment()
    line = env.new_line(components=[env.new('d', 'Drift', length=4.)])
    for name, radius, z in [('small', .02, 0.), ('large', .04, 2.0000001)]:
        profile = env.aperture.new_profile(name, 'Circle', radius=radius)
        pipe = env.aperture.new_pipe(name, positions=[profile.place(at=s) for s in [0., 2.]])
        line.aperture.place_pipe(name, pipe, at='d', shift_z=z)
    s = 2.0000001
    ref = line.aperture.get_cross_sections_at_s([s]).cross_section[0, :-1]
    limit = line.aperture.get_limit_elements_at_s([s])[s]
    assert isinstance(limit, xt.LimitPolygon)
    np.testing.assert_array_equal(np.column_stack((limit.x_vertices, limit.y_vertices)), ref)


def test_oblique_circular_taper_in_unsorted_planes():
    env = xt.Environment()
    line = env.new_line(components=[env.new('drift', 'Drift', length=2.)])
    small = env.aperture.new_profile('small', 'Circle', radius=.02)
    env['exit_radius'] = .07
    large = env.aperture.new_profile('large', 'Circle', radius='exit_radius')
    pipe = env.aperture.new_pipe('taper', positions=[small.place(at=0.), large.place(at=2.)])
    tilt = .2
    line.aperture.place_pipe('installed', pipe, at='drift', rot_y_rad=tilt)
    with pytest.warns(UserWarning, match='not translated'):
        cuts = line.aperture.get_limit_elements_at_s([1.5, .5], names=['last', 'first'])
    # Intersect x_pipe**2 + y_pipe**2 = (r0 + slope*z_pipe)**2 with the survey plane.
    c, sn = np.cos(tilt), np.sin(tilt)
    slope = (.07 - .02) / 2
    d = c**2 - (slope*sn)**2
    for s, element in cuts.items():
        r = .02 + slope*s/c
        assert isinstance(element, xt.LimitEllipse)
        np.testing.assert_allclose([element.a, element.b, element.shift_x],
            [r*c/d, r*c/np.sqrt(d), s*np.tan(tilt) + r*slope*sn/d], atol=1e-12)


@pytest.mark.parametrize('prefer_native', [False, True])
def test_generated_limit_keeps_helper_placement_and_bindings(prefer_native):
    env = xt.Environment()
    env['gap'] = .03
    line = env.new_line(components=[env.new('drift', 'Drift', length=1.)])
    profile = env.aperture.new_profile('jaw', 'Rectangle', max_x='gap', max_y=.1)
    line.aperture.new_element_pipe('drift', profile)
    if prefer_native:
        line.aperture.get_limit_elements_at_s([.5], names=['limit'])
    else:
        with pytest.warns(UserWarning, match='not translated'):
            line.aperture.get_limit_elements_at_s([.5], names=['limit'], prefer_native=False)
    assert env['limit'].installed_pipe_name == 'drift'
    env['gap'] = .04
    if prefer_native:
        assert env['limit'].max_x == .04
    else:
        assert isinstance(env.element_dict['limit'], xt.LimitPolygon)
