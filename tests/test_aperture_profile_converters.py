import numpy as np
import pytest

from xtrack.aperture.profile_converters import profile_from_madx_aperture
from xtrack.aperture.structures import (
    Circle, Ellipse, Octagon, Racetrack, Rectangle, RectEllipse,
)


APERTURES = [
    ('circle', [0.1], Circle, {'radius': 0.1}),
    ('rectangle', [0.1, 0.2], Rectangle,
     {'half_width': 0.1, 'half_height': 0.2}),
    ('ellipse', [0.1, 0.2], Ellipse,
     {'half_major': 0.1, 'half_minor': 0.2}),
    ('rectellipse', [0.1, 0.2, 0.3, 0.4], RectEllipse,
     {'half_width': 0.1, 'half_height': 0.2,
      'half_major': 0.3, 'half_minor': 0.4}),
    ('racetrack', [0.1, 0.2, 0.3, 0.4], Racetrack,
     {'half_width': 0.1, 'half_height': 0.2,
      'half_major': 0.3, 'half_minor': 0.4}),
    ('octagon', [0.1, 0.1, np.pi / 8, 3 * np.pi / 8], Octagon,
     {'half_width': 0.1, 'half_height': 0.1, 'half_diagonal': 0.1}),
]


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
