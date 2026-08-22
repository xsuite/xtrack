"""Entry and exit transformations for Xtrack element misalignments.

This is a Python port of ``track_misalignment_entry_straight`` and
``track_misalignment_entry_curved``, together with their exit counterparts,
from ``track_misalignments.h``. The returned values use the same
MAD-X-compatible angle conventions as the C tracking code.
"""

from math import atan2, cos, hypot, sin
from typing import NamedTuple

import numpy as np


class TransformParameters(NamedTuple):
    """Six shift and rotation parameters for one side of an element."""

    shift_x: float
    shift_y: float
    shift_s: float
    rot_y_rad: float
    rot_x_rad: float
    rot_s_rad_no_frame: float


# Compatibility with the original entry-only version of this module.
EntryTransform = TransformParameters


def _misalignment_matrix(
    shift_x,
    shift_y,
    shift_s,
    rot_y_rad,
    rot_x_rad,
    rot_s_rad_no_frame,
):
    s_phi = sin(rot_x_rad)
    c_phi = cos(rot_x_rad)
    s_theta = sin(rot_y_rad)
    c_theta = cos(rot_y_rad)
    s_psi = sin(rot_s_rad_no_frame)
    c_psi = cos(rot_s_rad_no_frame)

    return np.array(
        [
            [
                -s_phi * s_psi * s_theta + c_psi * c_theta,
                -c_psi * s_phi * s_theta - c_theta * s_psi,
                c_phi * s_theta,
                shift_x,
            ],
            [c_phi * s_psi, c_phi * c_psi, s_phi, shift_y],
            [
                -c_theta * s_phi * s_psi - c_psi * s_theta,
                -c_psi * c_theta * s_phi + s_psi * s_theta,
                c_phi * c_theta,
                shift_s,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def _curved_part_matrix(part_angle, h, rot_s_rad):
    c_part = cos(part_angle)
    s_part = sin(part_angle)
    c_tilt = cos(rot_s_rad)
    s_tilt = sin(rot_s_rad)

    delta_x = (c_part - 1.0) * c_tilt / h
    delta_y = (c_part - 1.0) * s_tilt / h
    delta_s = s_part / h

    return np.array(
        [
            [
                (c_part - 1.0) * c_tilt**2 + 1.0,
                (c_part - 1.0) * c_tilt * s_tilt,
                -c_tilt * s_part,
                delta_x,
            ],
            [
                (c_part - 1.0) * c_tilt * s_tilt,
                (c_part - 1.0) * s_tilt**2 + 1.0,
                -s_part * s_tilt,
                delta_y,
            ],
            [c_tilt * s_part, s_part * s_tilt, c_part, delta_s],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def _rigid_affine_inverse(matrix):
    inverse = np.eye(4)
    inverse[:3, :3] = matrix[:3, :3].T
    inverse[:3, 3] = -matrix[:3, :3].T @ matrix[:3, 3]
    return inverse


def _parameters_from_matrix(matrix):
    return TransformParameters(
        shift_x=float(matrix[0, 3]),
        shift_y=float(matrix[1, 3]),
        shift_s=float(matrix[2, 3]),
        rot_y_rad=atan2(matrix[0, 2], matrix[2, 2]),
        rot_x_rad=atan2(matrix[1, 2], hypot(matrix[1, 0], matrix[1, 1])),
        rot_s_rad_no_frame=atan2(matrix[1, 0], matrix[1, 1]),
    )


def get_entry_transform_straight(
    shift_x=0.0,
    shift_y=0.0,
    shift_s=0.0,
    rot_y_rad=0.0,
    rot_x_rad=0.0,
    rot_s_rad_no_frame=0.0,
    rot_shift_anchor=0.0,
    length=0.0,
    rot_s_rad=0.0,
):
    """Return the entry-transform parameters for a straight element.

    Parameters have the same names and conventions as the corresponding
    Xtrack element fields. ``rot_shift_anchor`` is the longitudinal offset of
    the rotation anchor from the element entry, in metres.

    ``length`` is accepted for API consistency with the C function but does
    not affect a straight entry. The element-frame rotation ``rot_s_rad`` is
    also intentionally not part of this result. It remains a separate
    rotation, applied after the six returned transformations, as in
    ``track_misalignment_entry_straight``.

    Returns
    -------
    TransformParameters
        ``(shift_x, shift_y, shift_s, rot_y_rad, rot_x_rad,
        rot_s_rad_no_frame)`` in tracking order.
    """
    c_phi = cos(rot_x_rad)

    entry_shift_x = (
        shift_x - rot_shift_anchor * c_phi * sin(rot_y_rad)
    )
    entry_shift_y = shift_y - rot_shift_anchor * sin(rot_x_rad)
    entry_shift_s = (
        shift_s
        - rot_shift_anchor * (c_phi * cos(rot_y_rad) - 1.0)
    )

    return TransformParameters(
        shift_x=entry_shift_x,
        shift_y=entry_shift_y,
        shift_s=entry_shift_s,
        rot_y_rad=rot_y_rad,
        rot_x_rad=rot_x_rad,
        rot_s_rad_no_frame=rot_s_rad_no_frame,
    )


def get_entry_transform(
    shift_x=0.0,
    shift_y=0.0,
    shift_s=0.0,
    rot_y_rad=0.0,
    rot_x_rad=0.0,
    rot_s_rad_no_frame=0.0,
    rot_shift_anchor=0.0,
    length=0.0,
    angle=0.0,
    h=0.0,
    rot_s_rad=0.0,
):
    """Return the entry-transform parameters for a curved element.

    This implements the conjugation used by
    ``track_misalignment_entry_curved``::

        entry = first_part @ misalignment @ inverse(first_part)

    For a thick element (``length != 0``), the effective curvature is
    ``angle / length`` and the supplied ``h`` is ignored. For a thin element,
    ``h`` supplies the parent curvature.

    ``rot_s_rad`` describes the element frame and is used to propagate to the
    anchor, but remains separate from the returned
    ``rot_s_rad_no_frame``. It must be applied as its own rotation after the
    six returned transformations, matching the C tracking order.

    Returns
    -------
    TransformParameters
        ``(shift_x, shift_y, shift_s, rot_y_rad, rot_x_rad,
        rot_s_rad_no_frame)`` in tracking order.
    """
    # A non-bending thick element, or an ordinary thin element, follows the
    # straight code path in track_misalignments.h.
    if angle == 0.0 and (length != 0.0 or h == 0.0):
        return get_entry_transform_straight(
            shift_x=shift_x,
            shift_y=shift_y,
            shift_s=shift_s,
            rot_y_rad=rot_y_rad,
            rot_x_rad=rot_x_rad,
            rot_s_rad_no_frame=rot_s_rad_no_frame,
            rot_shift_anchor=rot_shift_anchor,
            length=length,
            rot_s_rad=rot_s_rad,
        )

    misalignment_matrix = _misalignment_matrix(
        shift_x,
        shift_y,
        shift_s,
        rot_y_rad,
        rot_x_rad,
        rot_s_rad_no_frame,
    )

    if length != 0.0:
        h = angle / length
    if h == 0.0:
        raise ValueError(
            "A curved entry transformation needs nonzero curvature: provide "
            "nonzero angle and length, or provide h for a thin element."
        )

    part_angle = rot_shift_anchor * h
    matrix_first_part = _curved_part_matrix(part_angle, h, rot_s_rad)
    inverse_first_part = _rigid_affine_inverse(matrix_first_part)

    misaligned_entry = (
        matrix_first_part @ misalignment_matrix @ inverse_first_part
    )

    return _parameters_from_matrix(misaligned_entry)


def get_exit_transform_straight(
    shift_x=0.0,
    shift_y=0.0,
    shift_s=0.0,
    rot_y_rad=0.0,
    rot_x_rad=0.0,
    rot_s_rad_no_frame=0.0,
    rot_shift_anchor=0.0,
    length=0.0,
    rot_s_rad=0.0,
):
    """Return the exit-transform parameters for a straight element.

    ``rot_s_rad`` remains separate from the returned values. At the exit, its
    inverse, ``-rot_s_rad``, is applied first. The remaining transformations
    are then applied in inverse order: returned s, x, and y rotations, followed
    by the returned longitudinal and transverse shifts. This is the order used
    by ``track_misalignment_exit_straight``.

    Returns
    -------
    TransformParameters
        The three exit shifts and three signed exit rotations. In particular,
        the returned rotations are the negatives of the corresponding input
        misalignment rotations.
    """
    c_phi = cos(rot_x_rad)
    negative_part_length = rot_shift_anchor - length

    exit_shift_x = (
        negative_part_length * c_phi * sin(rot_y_rad) - shift_x
    )
    exit_shift_y = negative_part_length * sin(rot_x_rad) - shift_y
    exit_shift_s = (
        negative_part_length * (c_phi * cos(rot_y_rad) - 1.0) - shift_s
    )

    return TransformParameters(
        shift_x=exit_shift_x,
        shift_y=exit_shift_y,
        shift_s=exit_shift_s,
        rot_y_rad=-rot_y_rad,
        rot_x_rad=-rot_x_rad,
        rot_s_rad_no_frame=-rot_s_rad_no_frame,
    )


def get_exit_transform(
    shift_x=0.0,
    shift_y=0.0,
    shift_s=0.0,
    rot_y_rad=0.0,
    rot_x_rad=0.0,
    rot_s_rad_no_frame=0.0,
    rot_shift_anchor=0.0,
    length=0.0,
    angle=0.0,
    h=0.0,
    rot_s_rad=0.0,
):
    """Return the exit-transform parameters for a curved element.

    This implements the realignment conjugation used by
    ``track_misalignment_exit_curved``::

        realign = inverse(second_part) @ inverse(misalignment) @ second_part

    For a thick element (``length != 0``), the effective curvature is
    ``angle / length`` and the supplied ``h`` is ignored. For a thin element,
    ``h`` supplies the parent curvature. Straight cases are dispatched to
    :func:`get_exit_transform_straight`.

    ``rot_s_rad`` remains separate. Its inverse, ``-rot_s_rad``, is applied
    before the six returned transformations. Those six are applied in the
    usual shift-x/y, shift-s, rotate-y, rotate-x, rotate-s order.

    Returns
    -------
    TransformParameters
        ``(shift_x, shift_y, shift_s, rot_y_rad, rot_x_rad,
        rot_s_rad_no_frame)`` for the exit.
    """
    if angle == 0.0 and (length != 0.0 or h == 0.0):
        return get_exit_transform_straight(
            shift_x=shift_x,
            shift_y=shift_y,
            shift_s=shift_s,
            rot_y_rad=rot_y_rad,
            rot_x_rad=rot_x_rad,
            rot_s_rad_no_frame=rot_s_rad_no_frame,
            rot_shift_anchor=rot_shift_anchor,
            length=length,
            rot_s_rad=rot_s_rad,
        )

    misalignment_matrix = _misalignment_matrix(
        shift_x,
        shift_y,
        shift_s,
        rot_y_rad,
        rot_x_rad,
        rot_s_rad_no_frame,
    )
    inverse_misalignment = _rigid_affine_inverse(misalignment_matrix)

    if length != 0.0:
        h = angle / length
    if h == 0.0:
        raise ValueError(
            "A curved exit transformation needs nonzero curvature: provide "
            "nonzero angle and length, or provide h for a thin element."
        )

    part_angle = angle - h * rot_shift_anchor
    matrix_second_part = _curved_part_matrix(part_angle, h, rot_s_rad)
    inverse_second_part = _rigid_affine_inverse(matrix_second_part)

    realign = (
        inverse_second_part @ inverse_misalignment @ matrix_second_part
    )
    return _parameters_from_matrix(realign)
