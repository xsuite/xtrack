from typing import NamedTuple

import numpy as np


class Transform(NamedTuple):
    shift_x: float
    shift_y: float
    shift_z: float
    rot_y: float
    rot_x: float
    rot_z: float


def transform_matrix(shift_x=0, shift_y=0, shift_z=0, rot_y=0, rot_x=0, rot_z=0):
    """Generate a 3D transformation matrix.

    Parameters
    ----------
    shift_x, shift_y, shift_z : float
        Shifts in x, y, and z directions
    rot_y : float
        Rotation around the y-axis (positive s to x) in radians (MAD-X theta)
    rot_x
        Rotation around the x-axis (positive s to y) in radians (MAD-X phi)
    rot_z
        Rotation around the z-axis (positive y to x) in radians (MAD-X psi)
    """
    s_phi, c_phi = np.sin(rot_x), np.cos(rot_x)
    s_theta, c_theta = np.sin(rot_y), np.cos(rot_y)
    s_psi, c_psi = np.sin(rot_z), np.cos(rot_z)
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

        R = Ry(rot_y) @ Rx(rot_x) @ Rz(rot_z)

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

    # Optional cleanup if numerical noise is present
    # Project to the nearest proper rotation matrix
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    # Extract translation directly
    shift_x, shift_y, shift_z = t

    # Extract angles for R = Ry(rot_y) @ Rx(rot_x) @ Rz(rot_z)
    rot_x = np.arcsin(np.clip(R[1, 2], -1.0, 1.0))
    rot_y = np.arctan2(R[0, 2], R[2, 2])
    rot_z = np.arctan2(R[1, 0], R[1, 1])

    return Transform(
        shift_x=float(shift_x),
        shift_y=float(shift_y),
        shift_z=float(shift_z),
        rot_y=float(rot_y),
        rot_x=float(rot_x),
        rot_z=float(rot_z),
    )
