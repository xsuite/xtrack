from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from xtrack import Frame


@dataclass
class Misalignment:
    dtheta: float
    dphi: float
    dpsi: float
    dx: float
    dy: float
    ds: float

    def apply_to_element(self, element):
        if (hasattr(element, 'rbend_model')
                and element.rbend_model == 'straight-body'):
            raise ValueError(
                'straight-body rbends not yet supported for misalignment '
                'application')
        element.rot_shift_anchor = 0.  # Defined at the entrance
        element.rot_y_rad = self.dtheta
        element.rot_x_rad = self.dphi
        element.rot_s_rad_no_frame = self.dpsi - element.rot_s_rad
        element.shift_x = self.dx
        element.shift_y = self.dy
        element.shift_s = self.ds


def clear_element_misalignments(element):
    element.rot_x_rad = 0
    element.rot_y_rad = 0
    element.rot_s_rad_no_frame = 0
    element.shift_x = 0
    element.shift_y = 0
    element.shift_s = 0
    element.rot_shift_anchor = 0


def misalignment_from_absolute_position(
        XYZ_elem_start, E_elem_start, XYZ_ref_start, E_ref_start,
        rbend_angle=None):
    """Infer MAD-X misalignments from absolute entrance position and frame.

    For an RBend, ``rbend_angle`` applies the half-angle transformation from
    its entrance frame to the frame used by the MAD-X misalignment convention.
    """
    frame_elem_start = Frame.from_survey(XYZ_elem_start, E_elem_start)

    if rbend_angle is not None:
        frame_elem_start.rotate_y(rbend_angle / 2)

    frame_ref_start = Frame.from_survey(XYZ_ref_start, E_ref_start)

    A = np.linalg.inv(frame_ref_start.matrix) @ frame_elem_start.matrix

    theta = np.arctan2(A[0, 2], A[2, 2])
    phi = np.arctan2(A[1, 2], np.sqrt(A[1, 0]**2 + A[1, 1]**2))
    psi = np.arctan2(A[1, 0], A[1, 1])
    dx = A[0, 3]
    dy = A[1, 3]
    ds = A[2, 3]

    return Misalignment(
        dtheta=theta, dphi=phi, dpsi=psi, dx=dx, dy=dy, ds=ds)


def rst_from_reference_start(
        XYZ_ref_start, E_ref_start, rot_s_rad, angle):
    frame_rst_start = Frame.from_survey(XYZ_ref_start, E_ref_start)
    frame_rst_start.rotate_s(rot_s_rad)
    frame_rst_start.rotate_y(-angle / 2)

    # S is along the chord, T is normal to the curvature plane, and R = S x T.
    es = frame_rst_start.ez
    et = frame_rst_start.ey
    er = np.cross(es, et)

    E_rst_start = np.column_stack((er, es, et))
    return frame_rst_start.XYZ.copy(), E_rst_start


def rst_start_end_offsets_from_positions(
        XYZ_rst_start, E_rst_start, XYZ_elem_start, XYZ_elem_end):
    displacement_start = XYZ_elem_start - XYZ_rst_start
    displacement_end = XYZ_elem_end - XYZ_rst_start
    offset_start_rst = E_rst_start.T @ displacement_start
    offset_end_rst = E_rst_start.T @ displacement_end
    return offset_start_rst, offset_end_rst


def rst_start_end_offsets_from_parameters(element, length):
    angle = getattr(element, 'angle', 0.0)

    frame_tilt = Frame().rotate_s(element.rot_s_rad)
    rotation_tilt = frame_tilt.E_matrix

    frame_half_angle = Frame().rotate_y(-angle / 2)
    rotation_half_angle = frame_half_angle.E_matrix

    frame_misalignment = Frame()
    frame_misalignment.rotate_y(element.rot_y_rad)
    frame_misalignment.rotate_x(-element.rot_x_rad)
    frame_misalignment.rotate_s(element.rot_s_rad_no_frame)
    rotation_misalignment = frame_misalignment.E_matrix

    # E_rst = E_ref @ rotation_tilt @ rotation_half_angle @ xys_from_rst.
    # The columns of xys_from_rst are R=-x, S=s, T=y, expressed in the
    # tilted chord frame.
    xys_from_rst = np.array([
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ])
    rst_from_xys = (
        xys_from_rst.T
        @ rotation_half_angle.T
        @ rotation_tilt.T
    )

    # The screenshot uses (DX, DS, DY). Xtrack uses (x, y, s), hence the
    # corresponding vectors below are (DX, DY, DS) and (0, 0, l_E).
    displacement_E_xys = np.array([
        element.shift_x,
        element.shift_y,
        element.shift_s,
    ])
    body_chord_xys = np.array([0.0, 0.0, length])

    b_E = rst_from_xys @ displacement_E_xys
    b_S = rst_from_xys @ (
        displacement_E_xys
        + rotation_misalignment
        @ rotation_tilt
        @ rotation_half_angle
        @ body_chord_xys
    )
    return b_E, b_S


def plot_exz(rotation_matrix, point, length=0.5, color='k'):
    """Plot the local x and z directions in the global Z-X plane."""
    if length <= 0:
        raise ValueError('length must be positive')

    rotation_matrix = np.asarray(rotation_matrix)
    point = np.asarray(point)
    ax = plt.gca()
    arrows = []

    for axis_index in (0, 2):
        direction = rotation_matrix[:, axis_index]
        delta_z = length * direction[2]
        delta_x = length * direction[0]
        projected_length = np.hypot(delta_z, delta_x)

        arrows.append(ax.arrow(
            point[2], point[0], delta_z, delta_x,
            width=0.025 * projected_length,
            head_width=0.15 * projected_length,
            head_length=0.25 * projected_length,
            length_includes_head=True,
            color=color,
        ))

    return arrows


def plot_exy(rotation_matrix, point, length=0.5, color='k'):
    """Plot the local x and y directions in the global X-Y plane."""
    if length <= 0:
        raise ValueError('length must be positive')

    rotation_matrix = np.asarray(rotation_matrix)
    point = np.asarray(point)
    ax = plt.gca()
    arrows = []

    for axis_index in (0, 1):
        direction = rotation_matrix[:, axis_index]
        delta_x = length * direction[0]
        delta_y = length * direction[1]
        projected_length = np.hypot(delta_x, delta_y)

        arrows.append(ax.arrow(
            point[0], point[1], delta_x, delta_y,
            width=0.025 * projected_length,
            head_width=0.15 * projected_length,
            head_length=0.25 * projected_length,
            length_includes_head=True,
            color=color,
        ))

    return arrows
