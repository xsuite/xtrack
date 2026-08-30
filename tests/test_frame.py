import numpy as np
import pytest

import xtrack as xt
import xtrack.beam_elements as beam_elements


def _reference_advance(XYZ, E_matrix, *, length=0, angle=0, tilt=0,
                       shift_x=0, shift_y=0,
                       rot_x=0, rot_y=0, rot_s=0):
    """Independent copy of the pre-refactor survey formulas."""
    if shift_x != 0 or shift_y != 0:
        return E_matrix @ np.array([shift_x, shift_y, 0]) + XYZ, E_matrix

    if rot_x != 0:
        c = np.cos(-rot_x)
        s = np.sin(-rot_x)
        rotation = np.array([[1, 0, 0], [0, c, s], [0, -s, c]])
        return XYZ, E_matrix @ rotation

    if rot_y != 0:
        c = np.cos(rot_y)
        s = np.sin(rot_y)
        rotation = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
        return XYZ, E_matrix @ rotation

    if rot_s != 0:
        c = np.cos(rot_s)
        s = np.sin(rot_s)
        rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        return XYZ, E_matrix @ rotation

    if angle == 0:
        return E_matrix @ np.array([0, 0, length]) + XYZ, E_matrix

    c = np.cos(angle)
    s = np.sin(angle)
    ct = np.cos(tilt)
    st = np.sin(tilt)
    rho = length / angle
    displacement = np.array([rho * (c - 1), 0, rho * s])
    rotation = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
    tilt_rotation = np.array([[ct, -st, 0], [st, ct, 0], [0, 0, 1]])
    inverse_tilt = np.array([[ct, st, 0], [-st, ct, 0], [0, 0, 1]])
    return (
        E_matrix @ (tilt_rotation @ displacement) + XYZ,
        E_matrix @ (tilt_rotation @ rotation @ inverse_tilt),
    )


@pytest.mark.parametrize('kwargs', [
    {'length': 1.7},
    {'shift_x': 0.2, 'shift_y': -0.3},
    {'rot_x': 0.23},
    {'rot_y': -0.31},
    {'rot_s': 0.47},
    {'length': 2.3, 'angle': 0.37},
    {'length': 2.3, 'angle': -0.37, 'tilt': 0.61},
    {'length': 0, 'angle': 0.37, 'tilt': -0.42},
])
def test_frame_matches_previous_survey_formulas(kwargs):
    frame = xt.Frame.from_survey_angles(
        X=1.2, Y=-0.7, Z=3.1, theta=0.21, phi=-0.17, psi=0.33)
    expected_XYZ, expected_E_matrix = _reference_advance(
        frame.XYZ.copy(), frame.E_matrix.copy(), **kwargs)

    if kwargs.get('shift_x', 0) != 0 or kwargs.get('shift_y', 0) != 0:
        frame.trans_x(kwargs.get('shift_x', 0))
        frame.trans_y(kwargs.get('shift_y', 0))
    elif kwargs.get('rot_x', 0) != 0:
        frame.rot_x(kwargs['rot_x'])
    elif kwargs.get('rot_y', 0) != 0:
        frame.rot_y(-kwargs['rot_y'])
    elif kwargs.get('rot_s', 0) != 0:
        frame.rot_s(kwargs['rot_s'])
    else:
        frame.arc(
            length=kwargs.get('length', 0),
            angle=kwargs.get('angle', 0),
            tilt=kwargs.get('tilt', 0),
        )

    np.testing.assert_allclose(frame.XYZ, expected_XYZ, atol=1e-15, rtol=0)
    np.testing.assert_allclose(
        frame.E_matrix, expected_E_matrix, atol=1e-15, rtol=0)


def test_frame_uses_survey_names():
    frame = xt.Frame.from_survey(
        XYZ=[1, 2, 3],
        E_matrix=np.eye(3),
    )

    np.testing.assert_array_equal(frame.XYZ, [1, 2, 3])
    np.testing.assert_array_equal(frame.E_matrix, np.eye(3))
    assert not hasattr(frame, 'xyz')
    assert not hasattr(frame, 'rotation')


def test_frame_exposes_survey_vectors_as_writable_views():
    frame = xt.Frame()

    assert np.shares_memory(frame.ex, frame.matrix)
    assert np.shares_memory(frame.ey, frame.matrix)
    assert np.shares_memory(frame.ez, frame.matrix)

    frame.ex = [0, 1, 0]
    frame.ey[:] = [0, 0, 1]
    frame.ez = [1, 0, 0]

    np.testing.assert_array_equal(frame.E_matrix, [
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
    ])


def test_frame_exposes_survey_angles_as_read_only_properties():
    expected = dict(theta=0.21, phi=-0.17, psi=0.33)
    frame = xt.Frame.from_survey_angles(**expected)

    assert frame.theta == pytest.approx(expected['theta'], abs=1e-15)
    assert frame.phi == pytest.approx(expected['phi'], abs=1e-15)
    assert frame.psi == pytest.approx(expected['psi'], abs=1e-15)

    with pytest.raises(AttributeError):
        frame.theta = 0


def test_frame_is_mutable_and_chainable():
    frame = xt.Frame()
    returned = frame.trans_x(1).rot_s(np.pi / 2).trans_x(2)

    assert returned is frame
    np.testing.assert_allclose(frame.XYZ, [1, 2, 0], atol=1e-15, rtol=0)


def test_frame_copy_and_arc_backtrack():
    initial = xt.Frame.from_survey_angles(
        X=0.4, Y=-1.2, Z=2.1, theta=0.2, phi=-0.3, psi=0.1)
    frame = initial.copy()
    frame.arc(length=1.8, angle=0.43, tilt=-0.37)
    frame.arc(length=-1.8, angle=-0.43, tilt=-0.37)

    np.testing.assert_allclose(frame.matrix, initial.matrix, atol=2e-15, rtol=0)
    assert not np.shares_memory(initial.matrix, initial.copy().matrix)


def test_frame_arc_conveniences():
    horizontal = xt.Frame().arc_x(length=1.2, angle=0.3)
    generic_horizontal = xt.Frame().arc(length=1.2, angle=0.3, tilt=0)
    vertical = xt.Frame().arc_y(length=1.2, angle=0.3)
    generic_vertical = xt.Frame().arc(
        length=1.2, angle=0.3, tilt=np.pi / 2)

    np.testing.assert_array_equal(horizontal.matrix, generic_horizontal.matrix)
    np.testing.assert_array_equal(vertical.matrix, generic_vertical.matrix)


def test_line_survey_uses_track_frame_hook(monkeypatch):
    seen_frames = []

    def track_frame(self, frame, backtrack=False):
        seen_frames.append(frame)
        frame.trans_x(-1 if backtrack else 1)

    monkeypatch.setattr(
        xt.Marker, 'track_frame', track_frame, raising=False)

    survey = xt.Line(elements=[xt.Marker()]).survey()

    assert len(seen_frames) == 1
    assert isinstance(seen_frames[0], xt.Frame)
    np.testing.assert_array_equal(survey.XYZ[-1], [1, 0, 0])


@pytest.mark.parametrize('name', [
    'advance_bend',
    'advance_rotation',
    'advance_drift',
    'advance_element',
    'get_E_from_angles',
    'get_angles_from_w',
    'compute_survey',
])
def test_procedural_survey_helpers_are_removed(name):
    assert not hasattr(xt.survey, name)


def test_survey_advance_element_export_is_removed():
    assert not hasattr(beam_elements, 'survey_advance_element')
