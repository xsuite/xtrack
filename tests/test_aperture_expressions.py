import pickle

import numpy as np
import pytest

import xtrack as xt
from xtrack.aperture.structures import Circle, LineApertureModel


def make_model():
    env = xt.Environment()
    env['setting'] = .015
    env['gap'] = '2 * setting'
    env['offset'] = 0.
    profile = env.aperture.new_profile('jaw', 'Rectangle', max_x='gap', max_y=.2)
    other = env.aperture.new_profile('round', 'Circle', radius=.1)
    env.aperture.new_profile('unused', 'Circle', radius=.3)
    pipe = env.aperture.new_pipe('jaw_pipe', positions=[
        profile.place(at=0., shift_x='offset'), profile.place(at=1., shift_x='offset')])
    pipe2 = env.aperture.new_pipe('round_pipe', positions=[other.place(), other.place(at=1.)])
    line = env.new_line(name='line', components=[
        env.new('d1', 'Drift', length=1.), env.new('gap_drift', 'Drift', length=1.),
        env.new('d2', 'Drift', length=1.)])
    line.aperture.place_pipe('jaw', pipe, at='d1')
    line.aperture.place_pipe('round', pipe2, at='d2')
    return env, line


def test_views_and_expressions_follow_storage():
    env, line = make_model()
    shape = env.aperture.profiles['jaw'].shape
    position = env.aperture.pipes['jaw_pipe'][1]
    assert shape.max_x == .03
    for action, value in ((line.aperture.build_model, .02),
                          (env.aperture.discard_model, .025),
                          (line.aperture.build_model, .035)):
        action()
        env['setting'] = value
        env['offset'] = .005
        assert shape.max_x == 2 * value
        assert position.shift_x == .005
        np.testing.assert_allclose(line.aperture.get_extents_at_s([.5]).max_x,
                                   [2 * value + .005], atol=1e-12)
    shape.max_x = .07
    env['setting'] = .1
    assert shape.max_x == .07
    assert env.aperture.ref.profiles['jaw'].shape.max_x.xdeps.expr is None


@pytest.mark.parametrize('built', [False, True])
@pytest.mark.parametrize('method', ['environment_copy', 'environment_json',
                                   'line_copy', 'line_json', 'aperture_json', 'pickle'])
def test_copy_and_serialization(built, method, tmp_path):
    env, line = make_model()
    if built:
        line.aperture.build_model()
    if method == 'environment_copy':
        restored = env.copy(with_progress=False)
    elif method == 'environment_json':
        env.to_json(tmp_path / 'environment.json', include_aperture=True)
        restored = xt.Environment.from_json(tmp_path / 'environment.json', with_progress=False)
    elif method == 'line_copy':
        restored = line.copy().env
    elif method == 'line_json':
        line.to_json(tmp_path / 'line.json', include_aperture=True)
        restored = xt.Line.from_json(tmp_path / 'line.json', verbose=False).env
    elif method == 'pickle':
        restored = pickle.loads(pickle.dumps(env))
    else:
        restored = xt.Environment()
        restored['setting'] = .015
        restored['gap'] = '2 * setting'
        restored['offset'] = 0.
        env.aperture.to_json(tmp_path / 'aperture.json', include_line_models=False)
        restored.aperture.from_json(tmp_path / 'aperture.json')
    restored['setting'] = .04
    restored['offset'] = .01
    assert restored.aperture.profiles['jaw'].shape.max_x == .08
    assert restored.aperture.pipes['jaw_pipe'][1].shift_x == .01
    assert env.aperture.profiles['jaw'].shape.max_x == .03
    assert env.aperture.pipes['jaw_pipe'][1].shift_x == 0.
    if method == 'pickle':
        restored_line = restored.lines['line']
        np.testing.assert_allclose(restored_line.aperture.get_extents_at_s([.5]).max_x,
                                   [.09], atol=1e-12)
        restored.aperture.discard_model()
        assert not restored_line.aperture.is_built


def test_numeric_export_and_aperture_omission():
    env, _ = make_model()
    omitted = xt.Environment.from_dict(env.to_dict(), with_progress=False)
    omitted['setting'] = .05  # No dangling tasks targeting omitted apertures.
    assert not omitted.aperture.profiles.keys()
    numeric = xt.Environment.from_dict(
        env.to_dict(include_aperture=True, include_var_management=False), with_progress=False)
    numeric['gap'] = .1
    assert numeric.aperture.profiles['jaw'].shape.max_x == .03
    env['aperture.scale'] = 2
    env['scaled'] = env.ref['aperture.scale'] * env.ref['gap']
    xt.Environment.from_dict(env.to_dict(), with_progress=False)
    env['opening'] = 2 * env.aperture.ref.profiles['jaw'].shape.max_x
    with pytest.raises(ValueError, match='include_aperture=True'):
        env.to_dict()
    restored = env.copy(with_progress=False)
    restored['gap'] = .04
    assert restored['opening'] == .08


def test_selective_lazy_refresh(monkeypatch):
    env, line = make_model()
    line.aperture.build_model()
    survey = line.aperture.survey
    polygons = line.aperture._profile_polygons
    bounds = line.aperture._aperture_bounds
    calls = []
    original = LineApertureModel.build_profile_polygons

    def record(self, **kwargs):
        calls.append((np.array(kwargs['profile_mask']), np.array(kwargs['bounds_mask'])))
        return original(self, **kwargs)

    monkeypatch.setattr(LineApertureModel, 'build_profile_polygons', record)
    line.aperture.get_extents_at_s([.5])
    assert not calls
    env.aperture.profiles['unused'].shape.radius = .4
    line.aperture.get_extents_at_s([.5])
    assert not calls
    env['gap'] = .05
    assert not calls  # Updates don't eagerly run geometry kernels.
    line.aperture.get_poses_at_s([.5])  # Survey queries don't need aperture geometry.
    assert not calls
    np.testing.assert_allclose(line.aperture.get_extents_at_s([.5]).max_x, [.05], atol=1e-12)
    np.testing.assert_array_equal(calls[-1][0], [1, 0, 0])
    assert calls[-1][1].sum() == 2
    assert len(calls) == 1
    line.aperture.get_cross_sections_at_s([.5])
    line.aperture.get_bounds_table()
    line.aperture.get_pipe_table()
    assert len(calls) == 1
    env['offset'] = .01
    np.testing.assert_allclose(line.aperture.get_extents_at_s([.5]).max_x, [.06], atol=1e-12)
    assert not calls[-1][0].any()
    assert calls[-1][1].sum() == 2
    assert len(calls) == 2
    env.aperture.pipes['jaw_pipe'][0].tol_x = .001
    line.aperture.get_extents_at_s([.5])
    assert len(calls) == 2
    line.aperture.pipe_positions['jaw'].shift_x = .005
    np.testing.assert_allclose(line.aperture.get_extents_at_s([.5]).max_x, [.065], atol=1e-12)
    assert not calls[-1][0].any()
    assert calls[-1][1].sum() == 2
    assert line.aperture.survey is survey
    assert line.aperture._profile_polygons is polygons
    assert line.aperture._aperture_bounds is bounds

    env.aperture.pipes['jaw_pipe'][1].shift_x = .02
    line.aperture.get_bounds_table()
    assert not calls[-1][0].any()
    assert calls[-1][1].sum() == 1

    line.aperture.s_tol *= 2
    count = len(calls)
    line.aperture.get_bounds_table()
    assert len(calls) == count  # Only ordering/overlap checks depend on s_tol.

    env['gap'] = -.01
    count = len(calls)
    with pytest.raises(ValueError, match='Invalid aperture profile'):
        line.aperture.get_bounds_table()
    assert len(calls) == count
    env['gap'] = .04
    line.aperture.get_bounds_table()
    np.testing.assert_array_equal(calls[-1][0], [1, 0, 0])

    # A previously unused profile has no polygon, despite being in the packed
    # environment. Switching a position to it must build that polygon too.
    env.aperture.pipes['jaw_pipe'][1].profile_index = env.aperture.profile_index('unused')
    line.aperture.get_bounds_table()
    np.testing.assert_array_equal(calls[-1][0], [0, 0, 1])
    assert calls[-1][1].sum() == 1


def test_shared_profiles_refresh_each_line_on_demand():
    env, line = make_model()
    other = line.copy(shallow=True)
    line.aperture.build_model()
    other.aperture.build_model()
    env['gap'] = .06
    for current in (line, other):
        np.testing.assert_allclose(current.aperture.get_extents_at_s([.5]).max_x, [.06], atol=1e-12)


def test_pipe_references_and_position_identity():
    env, line = make_model()
    pipe = env.aperture.pipes['jaw_pipe']
    pipe.curvature = 'gap / 100'
    pipe[1].shift_s = '20 * gap'
    env.aperture.build_model()
    env['gap'] = .04
    assert pipe.curvature == .0004
    assert pipe[1].shift_s == .8
    env.aperture.discard_model()
    env['gap'] = .05
    assert pipe[1].shift_s == 1.


def test_clear_removes_expressions():
    env, _ = make_model()
    env.aperture.clear()
    env['gap'] = .1
    assert not any(lhs.startswith('aperture.') for lhs, _ in env.ref_manager.dump())


def test_polygon_vertex_view_and_expression():
    env = xt.Environment()
    env['x'] = .1
    profile = env.aperture.new_profile('polygon', 'Polygon',
        vertices=[[-.1, -.1], [.1, -.1], [.1, .1], [-.1, .1]])
    vertices = profile.shape.vertices
    vertices[1, 0] = env.ref['x']
    env.aperture.build_model()
    env['x'] = .2
    assert vertices[1, 0] == .2
    env.aperture.discard_model()
    env['x'] = .3
    assert vertices[1, 0] == .3
    copied = env.copy(with_progress=False)
    copied['x'] = .4
    assert copied.aperture.profiles['polygon'].shape.vertices[1, 0] == .4
    assert vertices[1, 0] == .3


def test_profile_replacement_clears_old_expressions():
    env, _ = make_model()
    profile = env.aperture.profiles['jaw']
    profile.shape = Circle(radius=.1)
    env['gap'] = .05
    assert profile.shape.radius == .1
    profile.shape.radius = 'gap'
    env['gap'] = .08
    assert profile.shape.radius == .08
