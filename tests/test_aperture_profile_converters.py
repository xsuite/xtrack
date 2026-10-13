import numpy as np
import pytest

import xtrack as xt

from xtrack.aperture.profile_converters import profile_from_madx_aperture
from xtrack.aperture.structures import (
    Circle, Ellipse, Octagon, Racetrack, Rectangle, RectEllipse,
)


APERTURES = [
    ('circle', [0.1], Circle, {'radius': 0.1}),
    ('rectangle', [0.1, 0.2], Rectangle,
     {'max_x': 0.1, 'max_y': 0.2}),
    ('ellipse', [0.1, 0.2], Ellipse,
     {'a': 0.1, 'b': 0.2}),
    ('rectellipse', [0.1, 0.2, 0.3, 0.4], RectEllipse,
     {'max_x': 0.1, 'max_y': 0.2,
      'a': 0.3, 'b': 0.4}),
    ('racetrack', [0.1, 0.2, 0.3, 0.4], Racetrack,
     {'max_x': 0.1, 'max_y': 0.2,
      'a': 0.3, 'b': 0.4}),
    ('octagon', [0.1, 0.1, np.pi / 8, 3 * np.pi / 8], Octagon,
     {'max_x': 0.1, 'max_y': 0.1, 'max_diagonal': 0.1}),
]


@pytest.mark.parametrize('shape,params,profile_type,fields', APERTURES)
def test_shape_parameters_survive_build_and_serialization(
        shape, params, profile_type, fields):
    env = xt.Environment()
    profile = env.aperture.new_profile('profile', profile_type, **fields)
    expected = profile.shape._to_dict()

    env.aperture.build_model()
    assert profile.shape._to_dict() == expected

    data = env.aperture.to_dict(include_line_models=False)
    assert data['profiles'][0]['shape'] == (profile_type.__name__, expected)

    restored = xt.Environment()
    restored.aperture.from_dict(data)
    assert restored.aperture.profiles['profile'].shape._to_dict() == expected

    env.aperture.discard_model()
    assert profile.shape._to_dict() == expected


@pytest.mark.parametrize('shape,params,profile_type,fields', APERTURES)
@pytest.mark.parametrize('container', [list, np.array])
@pytest.mark.parametrize('padding', [[], [0, 0]])
def test_madx_aperture_zero_padding(shape, params, profile_type, fields,
                                    container, padding):
    profile = profile_from_madx_aperture(shape, container(params + padding))
    assert isinstance(profile, profile_type)
    for field, expected in fields.items():
        assert getattr(profile, field) == pytest.approx(expected)

    zeros = [0] * len(params) + padding
    assert profile_from_madx_aperture(shape, container(zeros)) is None


@pytest.mark.parametrize('shape,params,profile_type,fields', APERTURES)
@pytest.mark.parametrize('container', [list, np.array])
@pytest.mark.parametrize('zero_core', [False, True])
def test_madx_aperture_rejects_extra_nonzero_parameters(
        shape, params, profile_type, fields, container, zero_core):
    if zero_core:
        params = [0] * len(params)
    with pytest.raises(ValueError, match='Extra non-zero parameters'):
        profile_from_madx_aperture(shape, container(params + [0, 0.5]))
