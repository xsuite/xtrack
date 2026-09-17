from types import SimpleNamespace

import numpy as np
import pytest

import xtrack as xt
from xtrack._temp import survey_utils as su


def _angle_difference(actual, expected):
    return np.arctan2(
        np.sin(actual - expected),
        np.cos(actual - expected),
    )


def _misalignment_values(misalignment):
    return np.array([
        misalignment.dtheta,
        misalignment.dphi,
        misalignment.dpsi,
        misalignment.shift_x,
        misalignment.shift_y,
        misalignment.shift_s,
    ])


def test_apply_and_clear_element_misalignment():
    element = SimpleNamespace(
        rbend_model='curved-body',
        rot_s_rad=0.17,
    )
    misalignment = su.Misalignment(
        dtheta=0.11,
        dphi=-0.23,
        dpsi=0.47,
        shift_x=1.2,
        shift_y=-0.7,
        shift_s=0.31,
    )

    misalignment.apply_to_element(element)

    assert element.rot_shift_anchor == 0
    assert element.rot_y_rad == misalignment.dtheta
    assert element.rot_x_rad == misalignment.dphi
    assert element.rot_s_rad_no_frame == pytest.approx(
        misalignment.dpsi - element.rot_s_rad)
    assert element.shift_x == misalignment.shift_x
    assert element.shift_y == misalignment.shift_y
    assert element.shift_s == misalignment.shift_s

    su.clear_element_misalignments(element)

    assert element.rot_shift_anchor == 0
    assert element.rot_y_rad == 0
    assert element.rot_x_rad == 0
    assert element.rot_s_rad_no_frame == 0
    assert element.shift_x == 0
    assert element.shift_y == 0
    assert element.shift_s == 0


def test_apply_misalignment_rejects_straight_body_rbend():
    element = SimpleNamespace(
        rbend_model='straight-body',
        rot_s_rad=0,
    )
    misalignment = su.Misalignment(0, 0, 0, 0, 0, 0)

    with pytest.raises(ValueError, match='straight-body rbends'):
        misalignment.apply_to_element(element)


def test_randomized_rst_misalignment_round_trips():
    rng = np.random.default_rng(20260831)

    for ii in range(300):
        reference_frame = xt.Frame.from_survey_angles(
            X=rng.uniform(-100, 100),
            Y=rng.uniform(-100, 100),
            Z=rng.uniform(-100, 100),
            theta=rng.uniform(-1, 1),
            phi=rng.uniform(-0.7, 0.7),
            psi=rng.uniform(-1, 1),
        )
        length = rng.uniform(0.05, 10)
        angle = 0 if ii % 10 == 0 else rng.uniform(-0.7, 0.7)
        tilt = rng.uniform(-np.pi, np.pi)
        element = SimpleNamespace(
            angle=angle,
            rot_s_rad=tilt,
            rot_y_rad=rng.uniform(-0.4, 0.4),
            rot_x_rad=rng.uniform(-0.4, 0.4),
            rot_s_rad_no_frame=rng.uniform(-0.4, 0.4),
            shift_x=rng.uniform(-2, 2),
            shift_y=rng.uniform(-2, 2),
            shift_s=rng.uniform(-2, 2),
        )

        expected_start_rst, expected_end_rst = (
            su.rst_start_end_offsets_from_parameters(element, length)
        )
        XYZ_rst_start, E_rst_start = su.rst_from_reference_start(
            XYZ_ref_start=reference_frame.XYZ,
            E_ref_start=reference_frame.E_matrix,
            rot_s_rad=tilt,
            angle=angle,
        )
        XYZ_elem_start = XYZ_rst_start + E_rst_start @ expected_start_rst
        XYZ_elem_end = XYZ_rst_start + E_rst_start @ expected_end_rst

        element_start_frame = reference_frame.copy()
        element_start_frame.translate_x(element.shift_x)
        element_start_frame.translate_y(element.shift_y)
        element_start_frame.translate_s(element.shift_s)
        element_start_frame.rotate_y(element.rot_y_rad)
        element_start_frame.rotate_x(-element.rot_x_rad)
        element_start_frame.rotate_s(element.rot_s_rad_no_frame)
        element_start_frame.rotate_s(element.rot_s_rad)

        rbend_angle = angle if ii % 2 else None
        if rbend_angle is not None:
            element_start_frame.rotate_y(-rbend_angle / 2)

        offset_start_rst, offset_end_rst, bgamma = (
            su.rst_start_end_offsets_tilt_from_positions(
                XYZ_rst_start=XYZ_rst_start,
                E_rst_start=E_rst_start,
                XYZ_elem_start=XYZ_elem_start,
                E_elem_start=element_start_frame.E_matrix,
                XYZ_elem_end=XYZ_elem_end,
                tilt=tilt,
                angle=angle,
                rbend_angle=rbend_angle,
            )
        )
        from_rst = su.misalignment_from_rst_offsets(
            offset_start_rst=offset_start_rst,
            offset_end_rst=offset_end_rst,
            bgamma=bgamma,
            tilt=tilt,
            angle=angle,
        )
        from_absolute = su.misalignment_from_absolute_position(
            XYZ_elem_start=XYZ_elem_start,
            E_elem_start=element_start_frame.E_matrix,
            XYZ_ref_start=reference_frame.XYZ,
            E_ref_start=reference_frame.E_matrix,
            rbend_angle=rbend_angle,
        )

        np.testing.assert_allclose(
            offset_start_rst, expected_start_rst, atol=5e-13, rtol=0)
        np.testing.assert_allclose(
            offset_end_rst, expected_end_rst, atol=5e-13, rtol=0)
        assert _angle_difference(
            bgamma, -element.rot_s_rad_no_frame) == pytest.approx(
                0, abs=5e-13)

        expected = np.array([
            element.rot_y_rad,
            element.rot_x_rad,
            element.rot_s_rad + element.rot_s_rad_no_frame,
            element.shift_x,
            element.shift_y,
            element.shift_s,
        ])
        for actual in (from_rst, from_absolute):
            actual_values = _misalignment_values(actual)
            np.testing.assert_allclose(
                actual_values[3:], expected[3:], atol=5e-13, rtol=0)
            for actual_angle, expected_angle in zip(
                    actual_values[:3], expected[:3]):
                assert _angle_difference(
                    actual_angle, expected_angle) == pytest.approx(
                        0, abs=5e-13)

        np.testing.assert_allclose(
            E_rst_start.T @ E_rst_start,
            np.eye(3), atol=5e-15, rtol=0)
        assert np.linalg.det(E_rst_start) == pytest.approx(1, abs=5e-15)


def test_misalignment_from_rst_offsets_rejects_degenerate_chords():
    with pytest.raises(ValueError, match='must define a chord'):
        su.misalignment_from_rst_offsets(
            offset_start_rst=np.zeros(3),
            offset_end_rst=np.zeros(3),
            bgamma=0,
        )

    with pytest.raises(ValueError, match='parallel to x'):
        su.misalignment_from_rst_offsets(
            offset_start_rst=np.zeros(3),
            offset_end_rst=np.array([1, 0, 0]),
            bgamma=0,
            angle=np.pi,
        )


@pytest.mark.parametrize('plot_function', [su.plot_exs, su.plot_exy])
def test_survey_frame_plot_helpers(plot_function):
    plt = pytest.importorskip('matplotlib.pyplot')

    plt.figure()
    arrows = plot_function(
        rotation_matrix=np.eye(3),
        point=np.array([1, 2, 3]),
        length=0.5,
        color='red',
    )

    assert len(arrows) == 2
    assert all(arrow.axes is plt.gca() for arrow in arrows)

    with pytest.raises(ValueError, match='length must be positive'):
        plot_function(np.eye(3), np.zeros(3), length=0)

    plt.close()


def test_rst_rigid_chord_preserves_the_chord_length():
    length = 2.5

    # A pure translation leaves the chord along S.
    chord = su.rst_rigid_chord([1e-3, -2e-3, 3e-3], [1e-3, 5e-3, 3e-3], length)
    np.testing.assert_allclose(chord, [0, length, 0], atol=5e-15, rtol=0)

    # A crab request tilts the chord, which keeps its length.
    displacement = 0.1
    chord = su.rst_rigid_chord(
        [0, 0, displacement], [0, 0, -displacement], length)
    expected_s = np.sqrt(length**2 - (2 * displacement)**2)
    np.testing.assert_allclose(
        chord, [0, expected_s, -2 * displacement], atol=5e-15, rtol=0)
    assert np.linalg.norm(chord) == pytest.approx(length, abs=5e-15)

    # The requested exit S displacement is not used.
    for exit_s in (-0.3, 0., 0.7):
        np.testing.assert_allclose(
            su.rst_rigid_chord([0, 0, 0], [2e-3, exit_s, -1e-3], length),
            su.rst_rigid_chord([0, 0, 0], [2e-3, 0., -1e-3], length),
            atol=0, rtol=0)


def test_rst_rigid_chord_rejects_oversized_transverse_displacement():
    with pytest.raises(ValueError, match='larger than the element chord'):
        su.rst_rigid_chord([0, 0, 0], [0, 0, 1.], length=0.5)


def test_misalignment_from_rst_displacements_round_trip():
    rng = np.random.default_rng(2024)

    for ii in range(32):
        length = rng.uniform(0.5, 5.)
        tilt = rng.uniform(-np.pi, np.pi)
        angle = 0. if ii % 3 == 0 else rng.uniform(-0.3, 0.3)
        element = SimpleNamespace(
            angle=angle,
            rot_s_rad=tilt,
            rot_y_rad=rng.uniform(-0.2, 0.2),
            rot_x_rad=rng.uniform(-0.2, 0.2),
            rot_s_rad_no_frame=rng.uniform(-0.2, 0.2),
            shift_x=rng.uniform(-0.05, 0.05),
            shift_y=rng.uniform(-0.05, 0.05),
            shift_s=rng.uniform(-0.05, 0.05),
        )

        # An exactly rigid motion, expressed as end-point displacements from
        # the nominal (0, 0, 0) and (0, length, 0).
        offset_start_rst, offset_end_rst = (
            su.rst_start_end_offsets_from_parameters(element, length))
        displ_start_rst = offset_start_rst
        displ_end_rst = offset_end_rst - np.array([0., length, 0.])

        misalignment = su.misalignment_from_rst_displacements(
            displ_start_rst=displ_start_rst,
            displ_end_rst=displ_end_rst,
            length=length,
            bgamma=-element.rot_s_rad_no_frame,
            tilt=tilt,
            angle=angle,
        )

        # A rigid request is honoured exactly: nothing has to be discarded.
        expected = np.array([
            element.rot_y_rad,
            element.rot_x_rad,
            element.rot_s_rad + element.rot_s_rad_no_frame,
            element.shift_x,
            element.shift_y,
            element.shift_s,
        ])
        actual = _misalignment_values(misalignment)
        np.testing.assert_allclose(
            actual[3:], expected[3:], atol=5e-13, rtol=0)
        for actual_angle, expected_angle in zip(actual[:3], expected[:3]):
            assert _angle_difference(
                actual_angle, expected_angle) == pytest.approx(0, abs=5e-13)


def test_misalignment_from_rst_displacements_keeps_the_element_rigid():
    # A crab request with no longitudinal component: a rigid element can only
    # answer it by rotating about its centre, which pulls the exit back along
    # the chord. The rotation follows from the preserved length.
    length = 2.99
    displacement = 0.1

    misalignment = su.misalignment_from_rst_displacements(
        displ_start_rst=[0., 0., displacement],
        displ_end_rst=[0., 0., -displacement],
        length=length,
        bgamma=0.,
    )

    chord_s = np.sqrt(length**2 - (2 * displacement)**2)
    assert misalignment.dphi == pytest.approx(
        -np.arctan(2 * displacement / chord_s), abs=5e-15)
    assert misalignment.dtheta == pytest.approx(0, abs=5e-15)
    np.testing.assert_allclose(
        [misalignment.shift_x, misalignment.shift_y, misalignment.shift_s],
        [0., displacement, 0.], atol=5e-15, rtol=0)


def _geode_body_frame(misalignment, tilt, angle):
    element = SimpleNamespace(rot_s_rad=tilt, angle=angle)
    misalignment.apply_to_element(element)
    body = (xt.Frame().rotate_y(element.rot_y_rad)
            .rotate_x(-element.rot_x_rad)
            .rotate_s(element.rot_s_rad_no_frame)
            .rotate_s(tilt).rotate_y(-angle/2))
    body.XYZ = [element.shift_x, element.shift_y, element.shift_s]
    return element, body


@pytest.mark.parametrize('tilt', [0., np.pi/2, -np.pi/2, .7])
@pytest.mark.parametrize('angle', [-.12, 0., .12])
def test_geode_crab_has_zero_roll_in_nominal_chord_frame(tilt, angle):
    start = np.array([.01, .02, .03])
    end = np.array([-.02, .8, -.01])  # Exit S is intentionally incompatible.
    length = 2.
    mis = su.misalignment_from_geode_displacements(
        start, end, length, bgamma=0., tilt=tilt, angle=angle)
    element, body = _geode_body_frame(mis, tilt, angle)
    recovered_start, recovered_end = su.rst_start_end_offsets_from_parameters(
        element, length)
    np.testing.assert_allclose(recovered_start, start, atol=1e-14, rtol=0)
    np.testing.assert_allclose(recovered_end[[0, 2]], end[[0, 2]], atol=1e-14, rtol=0)
    assert np.linalg.norm(recovered_end-recovered_start) == pytest.approx(length)

    nominal = xt.Frame().rotate_s(tilt).rotate_y(-angle/2)
    relative = nominal.inverse() @ body
    assert relative.psi == pytest.approx(0., abs=1e-14)


@pytest.mark.parametrize('angle', [-.12, .12])
@pytest.mark.parametrize('roll', [-.01, .01])
def test_geode_roll_moves_bend_exit_about_entrance_tangent(angle, roll):
    length = 2.
    mis = su.misalignment_from_geode_displacements(
        [0, 0, 0], [0, 0, 0], length, bgamma=roll, angle=angle)
    # GEODE does not introduce a compensating crab to keep the exit fixed.
    np.testing.assert_allclose(
        [mis.dtheta, mis.dphi, mis.dpsi], [0., 0., -roll], atol=1e-14, rtol=0)
    element, _ = _geode_body_frame(mis, tilt=0., angle=angle)
    start, end = su.rst_start_end_offsets_from_parameters(element, length)
    np.testing.assert_allclose(start, 0., atol=1e-14, rtol=0)
    assert end[2] == pytest.approx(length*np.sin(angle/2)*np.sin(roll), abs=1e-14)
    assert np.linalg.norm(end-start) == pytest.approx(length)


@pytest.mark.parametrize('tilt', [0., .99, np.pi/2])
def test_geode_combined_crab_roll_rotates_about_crabbed_tangent(tilt):
    start, end = [.003, .002, .001], [-.003, .002, -.001]
    length, angle, roll = 2., .12, .01
    no_roll = su.misalignment_from_geode_displacements(
        start, end, length, bgamma=0., tilt=tilt, angle=angle)
    with_roll = su.misalignment_from_geode_displacements(
        start, end, length, bgamma=roll, tilt=tilt, angle=angle)
    _, body_before = _geode_body_frame(no_roll, tilt, angle)
    _, body_after = _geode_body_frame(with_roll, tilt, angle)
    nominal = xt.Frame().rotate_s(tilt).rotate_y(-angle/2)
    tangent = body_before.E_matrix @ nominal.E_matrix.T @ [0., 0., 1.]
    # Rotate a socket about the displaced entrance tangent using Rodrigues'
    # formula, independently of the Euler parameter extraction.
    socket = np.array([.3, .6, 1.7])
    before = body_before.E_matrix @ socket
    expected = (before*np.cos(roll) - np.cross(tangent, before)*np.sin(roll)
                + tangent*np.dot(tangent, before)*(1-np.cos(roll)))
    np.testing.assert_allclose(body_after.E_matrix @ socket, expected, atol=1e-14, rtol=0)
    np.testing.assert_allclose(body_after.XYZ, body_before.XYZ, atol=1e-14, rtol=0)


def test_geode_straight_untilted_matches_existing_conversion():
    args = dict(displ_start_rst=[.002, .003, -.004],
                displ_end_rst=[-.001, .003, .007], length=1., bgamma=.02)
    new = su.misalignment_from_geode_displacements(**args)
    old = su.misalignment_from_rst_displacements(**args)
    np.testing.assert_allclose(_misalignment_values(new), _misalignment_values(old),
                               atol=1e-14, rtol=0)


@pytest.mark.parametrize('length', [0., -1.])
def test_geode_displacements_reject_nonpositive_length(length):
    with pytest.raises(ValueError, match='length must be positive'):
        su.misalignment_from_geode_displacements([0, 0, 0], [0, 0, 0], length, bgamma=0.)


def test_geode_displacements_reject_oversized_crab():
    with pytest.raises(ValueError, match='larger than the element chord'):
        su.misalignment_from_geode_displacements([0, 0, 0], [2, 0, 0], 1., bgamma=0.)
