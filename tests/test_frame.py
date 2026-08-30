import numpy as np
import pytest

import xtrack as xt
import xtrack.beam_elements as beam_elements


def _old_advance(v, w, *, length=0, angle=0, tilt=0,
                 shift_x=0, shift_y=0, rot_x=0, rot_y=0, rot_s=0):
    """Independent copy of the pre-refactor survey formulas."""
    if shift_x != 0 or shift_y != 0:
        return w @ np.array([shift_x, shift_y, 0]) + v, w

    if rot_x != 0:
        c = np.cos(-rot_x)
        s = np.sin(-rot_x)
        rotation = np.array([[1, 0, 0], [0, c, s], [0, -s, c]])
        return v, w @ rotation

    if rot_y != 0:
        c = np.cos(rot_y)
        s = np.sin(rot_y)
        rotation = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
        return v, w @ rotation

    if rot_s != 0:
        c = np.cos(rot_s)
        s = np.sin(rot_s)
        rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        return v, w @ rotation

    if angle == 0:
        return w @ np.array([0, 0, length]) + v, w

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
        w @ (tilt_rotation @ displacement) + v,
        w @ (tilt_rotation @ rotation @ inverse_tilt),
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
    frame = xt.Frame.from_xyz_angles(
        X=1.2, Y=-0.7, Z=3.1, theta=0.21, phi=-0.17, psi=0.33)
    expected_xyz, expected_rotation = _old_advance(
        frame.xyz.copy(), frame.rotation.copy(), **kwargs)

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

    np.testing.assert_allclose(frame.xyz, expected_xyz, atol=1e-15, rtol=0)
    np.testing.assert_allclose(
        frame.rotation, expected_rotation, atol=1e-15, rtol=0)


def test_frame_is_mutable_and_chainable():
    frame = xt.Frame()
    returned = frame.trans_x(1).rot_s(np.pi / 2).trans_x(2)

    assert returned is frame
    np.testing.assert_allclose(frame.xyz, [1, 2, 0], atol=1e-15, rtol=0)


def test_frame_copy_inverse_and_arc_backtrack():
    initial = xt.Frame.from_xyz_angles(
        X=0.4, Y=-1.2, Z=2.1, theta=0.2, phi=-0.3, psi=0.1)
    frame = initial.copy()
    frame.arc(length=1.8, angle=0.43, tilt=-0.37)
    frame.arc(length=-1.8, angle=-0.43, tilt=-0.37)

    np.testing.assert_allclose(frame.matrix, initial.matrix, atol=2e-15, rtol=0)
    np.testing.assert_allclose(
        (initial @ initial.inverse()).matrix, np.eye(4), atol=2e-15, rtol=0)
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
