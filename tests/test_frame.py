from dataclasses import asdict, is_dataclass

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


def _local_transform(displacement=None, rotation=None):
    transform = np.eye(4)
    if displacement is not None:
        transform[:3, 3] = displacement
    if rotation is not None:
        transform[:3, :3] = rotation
    return transform


def _rotation_x(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rotation_y(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rotation_s(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _arc_transform(length, angle, tilt):
    if angle == 0:
        return _local_transform(displacement=[0, 0, length])

    tilt_rotation = _rotation_s(tilt)
    displacement = tilt_rotation @ np.array([
        -0.5 * length * angle * np.sinc(angle / (2 * np.pi))**2,
        0,
        length * np.sinc(angle / np.pi),
    ])
    rotation = (
        tilt_rotation @ _rotation_y(-angle) @ tilt_rotation.T)
    return _local_transform(displacement=displacement, rotation=rotation)


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
        frame.translate_x(kwargs.get('shift_x', 0))
        frame.translate_y(kwargs.get('shift_y', 0))
    elif kwargs.get('rot_x', 0) != 0:
        frame.rotate_x(kwargs['rot_x'])
    elif kwargs.get('rot_y', 0) != 0:
        frame.rotate_y(-kwargs['rot_y'])
    elif kwargs.get('rot_s', 0) != 0:
        frame.rotate_s(kwargs['rot_s'])
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


def test_frame_repr():
    frame = xt.Frame.from_survey_angles(
        X=1.25,
        Y=-2.5,
        Z=3.75,
        theta=0.125,
        phi=-0.25,
        psi=0.5,
    )

    assert repr(frame) == (
        'Frame(X=1.25, Y=-2.5, Z=3.75, '
        'theta=0.125, phi=-0.25, psi=0.5)')


def test_frame_exposes_survey_vectors_as_writable_views():
    frame = xt.Frame()

    assert np.shares_memory(frame.ex, frame.matrix)
    assert np.shares_memory(frame.ey, frame.matrix)
    assert np.shares_memory(frame.es, frame.matrix)
    assert not hasattr(frame, 'ez')

    frame.ex = [0, 1, 0]
    frame.ey[:] = [0, 0, 1]
    frame.es = [1, 0, 0]

    np.testing.assert_array_equal(frame.E_matrix, [
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
    ])


def test_frame_exposes_xyz_scalar_properties():
    frame = xt.Frame.from_survey(
        XYZ=[1, 2, 3],
        E_matrix=np.eye(3),
    )

    assert frame.X == 1
    assert frame.Y == 2
    assert frame.Z == 3

    frame.X = 4
    frame.Y = 5
    frame.Z = 6

    np.testing.assert_array_equal(frame.XYZ, [4, 5, 6])


def test_frame_exposes_survey_angles_as_read_only_properties():
    expected = dict(theta=0.21, phi=-0.17, psi=0.33)
    frame = xt.Frame.from_survey_angles(**expected)

    assert frame.theta == pytest.approx(expected['theta'], abs=1e-15)
    assert frame.phi == pytest.approx(expected['phi'], abs=1e-15)
    assert frame.psi == pytest.approx(expected['psi'], abs=1e-15)

    with pytest.raises(AttributeError):
        frame.theta = 0


def test_frame_from_ccs():
    ccs = xt.CCSFrame(
        x=1.2, y=-0.7, z=3.1,
        theta_gon=13.37, phi=-0.17, psi=0.33)
    frame = xt.Frame.from_ccs(ccs)

    survey_to_ccs = np.array([
        [-1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
    ])
    theta_rad = ccs.theta_gon * np.pi / 200
    ct, cp, cs = np.cos([theta_rad, ccs.phi, ccs.psi])
    st, sp, ss = np.sin([theta_rad, ccs.phi, ccs.psi])
    ccs_E_matrix = (
        np.array([[ct, st, 0], [-st, ct, 0], [0, 0, 1]])
        @ np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        @ np.array([[cs, 0, -ss], [0, 1, 0], [ss, 0, cs]])
    )

    np.testing.assert_allclose(
        frame.XYZ,
        survey_to_ccs.T @ np.array([ccs.x, ccs.y, ccs.z]),
        atol=1e-15,
        rtol=0,
    )
    np.testing.assert_allclose(
        frame.E_matrix,
        survey_to_ccs.T @ ccs_E_matrix @ survey_to_ccs,
        atol=1e-15,
        rtol=0,
    )
    assert frame.to_ccs().theta_gon == pytest.approx(
        ccs.theta_gon, abs=1e-14)


def test_frame_to_ccs_round_trip():
    frame = xt.Frame.from_survey_angles(
        X=1.2, Y=-0.7, Z=3.1, theta=0.21, phi=-0.17, psi=0.33)

    ccs = frame.to_ccs()
    rebuilt = xt.Frame.from_ccs(ccs)

    assert is_dataclass(ccs)
    assert asdict(ccs).keys() == {
        'x', 'y', 'z', 'theta_gon', 'phi', 'psi'}
    np.testing.assert_allclose(rebuilt.matrix, frame.matrix, atol=1e-15, rtol=0)


def test_frame_is_mutable_and_chainable():
    frame = xt.Frame()
    returned = frame.translate_x(1).rotate_s(np.pi / 2).translate_x(2)

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


def test_frame_inverse_and_composition_are_non_mutating():
    parent = xt.Frame.from_survey_angles(
        X=0.4, Y=-1.2, Z=2.1, theta=0.2, phi=-0.3, psi=0.1)
    child = xt.Frame.from_survey_angles(
        X=-0.7, Y=0.3, Z=1.4, theta=-0.1, phi=0.25, psi=-0.4)
    parent_matrix = parent.matrix.copy()
    child_matrix = child.matrix.copy()

    composed = parent @ child
    inverse = parent.inverse()

    np.testing.assert_allclose(
        composed.matrix, parent_matrix @ child_matrix, atol=2e-15, rtol=0)
    np.testing.assert_allclose(
        (inverse @ parent).matrix, np.eye(4), atol=2e-15, rtol=0)
    np.testing.assert_array_equal(parent.matrix, parent_matrix)
    np.testing.assert_array_equal(child.matrix, child_matrix)


def test_frame_inverse_supports_general_affine_matrix():
    matrix = np.array([
        [2.0, 0.1, 0.0, 1.0],
        [0.0, 0.5, 0.2, -2.0],
        [0.0, 0.0, 1.5, 3.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    frame = xt.Frame(matrix)

    np.testing.assert_allclose(
        frame.inverse().matrix, np.linalg.inv(matrix), atol=1e-15, rtol=0)


def test_frame_arc_conveniences():
    horizontal = xt.Frame().arc_x(length=1.2, angle=0.3)
    generic_horizontal = xt.Frame().arc(length=1.2, angle=0.3, tilt=0)
    vertical = xt.Frame().arc_y(length=1.2, angle=0.3)
    generic_vertical = xt.Frame().arc(
        length=1.2, angle=0.3, tilt=np.pi / 2)

    np.testing.assert_array_equal(horizontal.matrix, generic_horizontal.matrix)
    np.testing.assert_array_equal(vertical.matrix, generic_vertical.matrix)


def test_frame_random_transformation_sequences():
    rng = np.random.default_rng(20260831)

    for _ in range(100):
        frame = xt.Frame.from_survey_angles(
            X=rng.uniform(-10, 10),
            Y=rng.uniform(-10, 10),
            Z=rng.uniform(-10, 10),
            theta=rng.uniform(-np.pi, np.pi),
            phi=rng.uniform(-np.pi / 2, np.pi / 2),
            psi=rng.uniform(-np.pi, np.pi),
        )
        expected = frame.matrix.copy()

        for _ in range(30):
            operation = rng.integers(7)
            if operation < 3:
                displacement = np.zeros(3)
                displacement[operation] = rng.uniform(-2, 2)
                (frame.translate_x, frame.translate_y, frame.translate_s)[
                    operation](displacement[operation])
                local_transform = _local_transform(
                    displacement=displacement)
            elif operation < 6:
                angle = rng.uniform(-np.pi, np.pi)
                rotation = (
                    _rotation_x, _rotation_y, _rotation_s)[operation - 3](
                        angle)
                (frame.rotate_x, frame.rotate_y, frame.rotate_s)[operation - 3](
                    angle)
                local_transform = _local_transform(rotation=rotation)
            else:
                length = rng.uniform(-3, 3)
                angle = rng.uniform(-1, 1)
                tilt = rng.uniform(-np.pi, np.pi)
                frame.arc(length=length, angle=angle, tilt=tilt)
                local_transform = _arc_transform(length, angle, tilt)

            expected = expected @ local_transform

        np.testing.assert_allclose(
            frame.matrix, expected, atol=2e-14, rtol=2e-14)
        np.testing.assert_allclose(
            frame.E_matrix.T @ frame.E_matrix,
            np.eye(3), atol=2e-14, rtol=0)
        assert np.linalg.det(frame.E_matrix) == pytest.approx(1, abs=2e-14)
        np.testing.assert_allclose(
            (frame.inverse() @ frame).matrix,
            np.eye(4), atol=2e-14, rtol=0)


@pytest.mark.parametrize('angle', [
    1e-3,
    1e-6,
    1e-9,
    1e-12,
    1e-15,
    -1e-9,
])
def test_frame_arc_is_accurate_for_small_angles(angle):
    length = 2.3
    tilt = 0.47
    frame = xt.Frame().arc(length=length, angle=angle, tilt=tilt)

    np.testing.assert_allclose(
        frame.matrix, _arc_transform(length, angle, tilt),
        atol=1e-15, rtol=1e-14)


@pytest.mark.parametrize('phi', [
    -np.pi / 2,
    -np.pi / 2 + 1e-12,
    np.pi / 2 - 1e-12,
    np.pi / 2,
])
def test_frame_ccs_round_trip_near_singularities(phi):
    ccs = xt.CCSFrame(
        x=1.2, y=-0.7, z=3.1,
        theta_gon=45.2, phi=phi, psi=-0.37,
    )
    frame = xt.Frame.from_ccs(ccs)
    rebuilt = xt.Frame.from_ccs(frame.to_ccs())

    np.testing.assert_allclose(
        rebuilt.matrix, frame.matrix, atol=2e-15, rtol=0)


def test_frame_validation_setters_and_no_op_transformations():
    with pytest.raises(ValueError, match='shape'):
        xt.Frame(np.eye(3))

    frame = xt.Frame()
    frame.XYZ = [1, 2, 3]
    frame.E_matrix = _rotation_x(0.2)
    frame.ey = [0, 1, 0]
    expected = frame.matrix.copy()

    assert frame.__matmul__(object()) is NotImplemented
    assert frame.translate_x(0) is frame
    assert frame.translate_y(0) is frame
    assert frame.translate_s(0) is frame
    assert frame.rotate_x(0) is frame
    assert frame.rotate_y(0) is frame
    assert frame.rotate_s(0) is frame
    np.testing.assert_array_equal(frame.matrix, expected)


def test_line_survey_uses_track_frame_hook(monkeypatch):
    seen_frames = []

    def track_frame(self, frame, backtrack=False):
        seen_frames.append(frame)
        frame.translate_x(-1 if backtrack else 1)

    monkeypatch.setattr(
        xt.Marker, 'track_frame', track_frame, raising=False)

    survey = xt.Line(elements=[xt.Marker()]).survey()

    assert len(seen_frames) == 1
    assert isinstance(seen_frames[0], xt.Frame)
    np.testing.assert_array_equal(survey.XYZ[-1], [1, 0, 0])


@pytest.mark.parametrize('element_class, kwargs', [
    (xt.Marker, {}),
    (xt.Drift, {'length': 1.7}),
    (xt.Bend, {'length': 2.3, 'angle': 0.37}),
    (xt.Bend, {'length': 2.3, 'angle': -0.37, 'rot_s_rad': 0.61}),
    (xt.Multipole, {'length': 0.4, 'hxl': 0.12}),
])
def test_track_frame_over_single_element_matches_line_survey(
        element_class, kwargs):
    element = element_class(**kwargs)
    initial = xt.Frame.from_survey_angles(
        X=1.2, Y=-0.7, Z=3.1, theta=0.21, phi=-0.17, psi=0.33)
    initial_matrix = initial.matrix.copy()
    expected = xt.Line(elements=[element]).survey(
        X0=initial.XYZ[0],
        Y0=initial.XYZ[1],
        Z0=initial.XYZ[2],
        theta0=initial.theta,
        phi0=initial.phi,
        psi0=initial.psi,
    )

    result = xt.track_frame(initial, element)

    assert result is initial
    np.testing.assert_allclose(
        initial.XYZ, expected.XYZ[-1], atol=2e-15, rtol=0)
    np.testing.assert_allclose(
        initial.E_matrix, expected.E_matrix[-1], atol=2e-15, rtol=0)

    xt.track_frame(initial, element, backtrack=True)
    np.testing.assert_allclose(
        initial.matrix, initial_matrix, atol=3e-15, rtol=0)


def test_track_frame_dispatches_element_hook_and_backtrack():
    calls = []

    class Element:
        def track_frame(self, frame, backtrack=False):
            calls.append((frame, backtrack))
            frame.translate_x(-2 if backtrack else 2)

    frame = xt.Frame()
    result = xt.track_frame(frame, Element(), backtrack=True)

    assert result is frame
    assert calls == [(frame, True)]
    np.testing.assert_array_equal(frame.XYZ, [-2, 0, 0])


def test_track_frame_over_sliced_elements_matches_line_survey():
    line = xt.Line(elements={
        'bend': xt.Bend(length=2.3, angle=0.37, rot_s_rad=0.61),
    })
    line.slice_thick_elements(
        slicing_strategies=[xt.Strategy(slicing=xt.Teapot(2))],
        with_progress=False,
    )
    survey = line.survey()
    frame = xt.Frame()

    for ii, element in enumerate(line._elements):
        xt.track_frame(frame, element)
        np.testing.assert_allclose(
            frame.XYZ, survey.XYZ[ii + 1], atol=2e-15, rtol=0)
        np.testing.assert_allclose(
            frame.E_matrix, survey.E_matrix[ii + 1], atol=2e-15, rtol=0)


def test_survey_table_get_frame_by_name_and_index():
    survey = xt.Line(elements={
        'drift': xt.Drift(length=1.2),
        'bend': xt.Bend(length=2.3, angle=0.37),
    }).survey()

    by_name = survey.get_frame('bend')
    by_index = survey.get_frame(1)

    np.testing.assert_array_equal(by_name.matrix, by_index.matrix)
    np.testing.assert_array_equal(by_name.XYZ, survey.XYZ[1])
    np.testing.assert_array_equal(by_name.E_matrix, survey.E_matrix[1])

    by_name.translate_x(1)
    assert not np.array_equal(by_name.XYZ, survey.XYZ[1])


def test_survey_table_get_element_frames():
    survey = xt.Line(elements={
        'drift': xt.Drift(length=1.2),
        'bend': xt.Bend(
            length=2.3,
            angle=0.37,
            shift_x=0.02,
            rot_s_rad=0.15,
        ),
    }).survey(include_element_frames=True)

    frames = survey.get_all_frames('bend')

    assert tuple(frames) == (
        'ref_start', 'ref_end', 'elem_start', 'elem_end')
    for which, frame in frames.items():
        expected = survey.get_frame('bend', which=which)
        np.testing.assert_array_equal(frame.matrix, expected.matrix)
        np.testing.assert_array_equal(
            frame.XYZ, survey[f'XYZ_{which}', 'bend'])
        np.testing.assert_array_equal(
            frame.E_matrix, survey[f'E_{which}', 'bend'])


def test_survey_table_get_element_frames_requires_optional_columns():
    survey = xt.Line(elements=[xt.Drift(length=1.2)]).survey()

    with pytest.raises(ValueError, match='include_element_frames=True'):
        survey.get_frame(0, which='elem_start')
    with pytest.raises(ValueError, match='include_element_frames=True'):
        survey.get_all_frames(0)


def test_survey_table_get_frame_rejects_invalid_selector():
    survey = xt.Line(elements=[xt.Marker()]).survey()

    with pytest.raises(ValueError, match="Invalid frame 'invalid'"):
        survey.get_frame(0, which='invalid')


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
