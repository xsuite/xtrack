from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


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


def misalignment_from_absolute_position(XYZ_elem_start, E_elem_start,
                                        XYZ_ref_start, E_ref_start):
    p_mat_elem_start = np.eye(4)
    p_mat_elem_start[:3, :3] = E_elem_start
    p_mat_elem_start[:3, 3] = XYZ_elem_start

    p_mat_ref_start = np.eye(4)
    p_mat_ref_start[:3, :3] = E_ref_start
    p_mat_ref_start[:3, 3] = XYZ_ref_start

    A = np.linalg.inv(p_mat_ref_start) @ p_mat_elem_start

    theta = np.arctan2(A[0, 2], A[2, 2])
    phi = np.arctan2(A[1, 2], np.sqrt(A[1, 0]**2 + A[1, 1]**2))
    psi = np.arctan2(A[1, 0], A[1, 1])
    dx = A[0, 3]
    dy = A[1, 3]
    ds = A[2, 3]

    return Misalignment(
        dtheta=theta, dphi=phi, dpsi=psi, dx=dx, dy=dy, ds=ds)


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
