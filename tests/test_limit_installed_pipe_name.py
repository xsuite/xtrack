import pickle

import pytest
import xtrack as xt


@pytest.mark.parametrize('cls, params', [
    (xt.LimitRect, {}), (xt.LimitEllipse, {}), (xt.LimitRectEllipse, {}),
    (xt.LimitRacetrack, {}), (xt.LimitPolygon, dict(x_vertices=[0, 1, 0],
                                                    y_vertices=[0, 0, 1])),
])
def test_installed_pipe_name(cls, params, tmp_path):
    assert cls.from_dict(cls(**params).to_dict()).installed_pipe_name is None
    env = xt.Environment()
    env.new('limit', cls, installed_pipe_name='vacuum.left::1', **params)
    line = env.new_line(components=['limit'])
    assert line['limit'].installed_pipe_name == 'vacuum.left::1'
    line['limit'].installed_pipe_name = 'longer.placement,following.placement'
    element = env.element_dict['limit']
    line.to_json(tmp_path / 'line.json')
    for copy in (element.copy(), line.copy().element_dict['limit'],
                 xt.Line.from_json(tmp_path / 'line.json').element_dict['limit'],
                 pickle.loads(pickle.dumps(element))):
        assert copy.installed_pipe_name == 'longer.placement,following.placement'


def test_sampled_limits_identify_placement_not_definition():
    env = xt.Environment()
    line = env.new_line(components=[env.new('d', 'Drift', length=6.)])
    profile = env.aperture.new_profile('bore', 'Circle', radius=.02)
    pipe = env.aperture.new_pipe('tube', positions=[profile.place(at=s) for s in [0., 1.]])
    for name, z in [('third', 4.), ('second', 2.), ('first', 1.)]:
        line.aperture.place_pipe(name, pipe, at='d', shift_z=z)
    limits = line.aperture.get_limit_elements_at_s([1.5, 2., 2.5, 3.5, 4.5])
    assert [el.installed_pipe_name for el in limits.values()] == [
        'first', 'first,second', 'second', 'second,third', 'third']
