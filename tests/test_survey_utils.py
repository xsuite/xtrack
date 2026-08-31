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


@pytest.mark.parametrize('plot_function', [su.plot_exz, su.plot_exy])
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
