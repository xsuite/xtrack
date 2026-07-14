from typing import NamedTuple, Literal

import numpy as np


Frame = Literal['curved', 'straight']


class Transform(NamedTuple):
    shift_x: float
    shift_y: float
    shift_z: float
    rot_y_rad: float
    rot_x_rad: float
    rot_z_rad: float


def transform_matrix(
    shift_x=0.,
    shift_y=0.,
    shift_z=0.,
    rot_y_rad=0.,
    rot_x_rad=0.,
    rot_z_rad=0.,
):
    """Generate a 3D transformation matrix.

    Parameters
    ----------
    shift_x, shift_y, shift_z : float
        Shifts in x, y, and z directions
    rot_y_rad : float
        Rotation around the y-axis (positive s to x) in radians (MAD-X theta)
    rot_x_rad
        Rotation around the x-axis (positive s to y) in radians (MAD-X phi)
    rot_z_rad
        Rotation around the z-axis (positive y to x) in radians (MAD-X psi)
    """
    s_phi, c_phi = np.sin(rot_x_rad), np.cos(rot_x_rad)
    s_theta, c_theta = np.sin(rot_y_rad), np.cos(rot_y_rad)
    s_psi, c_psi = np.sin(rot_z_rad), np.cos(rot_z_rad)
    matrix = np.array(
        [
            [
                -s_phi * s_psi * s_theta + c_psi * c_theta,
                -c_psi * s_phi * s_theta - c_theta * s_psi,
                c_phi * s_theta,
                shift_x,
            ],
            [
                c_phi * s_psi,
                c_phi * c_psi,
                s_phi,
                shift_y,
            ],
            [
                -c_theta * s_phi * s_psi - c_psi * s_theta,
                -c_psi * c_theta * s_phi + s_psi * s_theta,
                c_phi * c_theta,
                shift_z,
            ],
            [0, 0, 0, 1],
        ]
    )
    return matrix


def matrix_to_transform(matrix: np.ndarray) -> Transform:
    """Decompose a 4x4 homogeneous transform matrix into shifts and rotations.

    The rotations are applied in the following order:

        R = Ry(rot_y_rad) @ Rx(rot_x_rad) @ Rz(rot_z_rad)

    and translation taken from the last column.

    Parameters
    ----------
    matrix : np.ndarray
        4x4 homogeneous transform matrix

    Returns
    -------
    Transform
        The corresponding transform parameters.
    """
    matrix = np.asarray(matrix, dtype=float)

    if matrix.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 matrix, got shape {matrix.shape}")

    # Optional structural checks
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError("Matrix does not look like a homogeneous transform")

    R = matrix[:3, :3]
    t = matrix[:3, 3]

    # Extract translation directly
    shift_x, shift_y, shift_z = t

    # Extract angles for R = Ry(rot_y_rad) @ Rx(rot_x_rad) @ Rz(rot_z_rad)
    rot_x_rad = np.arcsin(R[1, 2])
    rot_y_rad = np.arctan2(R[0, 2], R[2, 2])
    rot_z_rad = np.arctan2(R[1, 0], R[1, 1])

    return Transform(
        shift_x=float(shift_x),
        shift_y=float(shift_y),
        shift_z=float(shift_z),
        rot_y_rad=float(rot_y_rad),
        rot_x_rad=float(rot_x_rad),
        rot_z_rad=float(rot_z_rad),
    )


def arc_matrix(length: float, angle: float, tilt: float) -> np.ndarray:
    """Generate a 4x4 homogeneous transformation matrix for an arc, given its parameters."""
    if abs(angle) < 1e-9:
        return transform_matrix(shift_z=length, rot_z_rad=tilt)

    ct = np.cos(tilt)
    st = np.sin(tilt)
    ca = np.cos(angle)
    sa = np.sin(angle)
    dx = length * (ca - 1) / angle
    ds = length * sa / angle
    return np.array(
        [
            [ct * ca, -st, -ct * sa, ct * dx],
            [st * ca, ct, -st * sa, st * dx],
            [sa, 0.0, ca, ds],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def poly2d_to_homogeneous(poly2d: np.ndarray) -> np.ndarray:
    """Convert a 2D polygon to 3D homogeneous coordinates."""
    num_points = poly2d.shape[0]
    poly_hom = np.column_stack((poly2d, np.zeros(num_points), np.ones(num_points))).T
    return poly_hom