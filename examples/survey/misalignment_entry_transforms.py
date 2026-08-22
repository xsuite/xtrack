"""Entry transformations corresponding to Xtrack element misalignments.

This is a Python port of ``track_misalignment_entry_straight`` and
``track_misalignment_entry_curved`` from ``track_misalignments.h``.  The
returned values use the same MAD-X-compatible angle conventions as the C
tracking code.
"""

from math import atan2, cos, hypot, sin
from typing import NamedTuple

import numpy as np


class EntryTransform(NamedTuple):
    """The six parameters applied before tracking an element body."""

    shift_x: float
    shift_y: float
    shift_s: float
    rot_y_rad: float
    rot_x_rad: float
    rot_s_rad_no_frame: float


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
    EntryTransform
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

    return EntryTransform(
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
    EntryTransform
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

    s_phi = sin(rot_x_rad)
    c_phi = cos(rot_x_rad)
    s_theta = sin(rot_y_rad)
    c_theta = cos(rot_y_rad)
    s_psi = sin(rot_s_rad_no_frame)
    c_psi = cos(rot_s_rad_no_frame)

    misalignment_matrix = np.array(
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

    if length != 0.0:
        h = angle / length
    if h == 0.0:
        raise ValueError(
            "A curved entry transformation needs nonzero curvature: provide "
            "nonzero angle and length, or provide h for a thin element."
        )

    part_angle = rot_shift_anchor * h
    c_part = cos(part_angle)
    s_part = sin(part_angle)
    c_tilt = cos(rot_s_rad)
    s_tilt = sin(rot_s_rad)

    delta_x = (c_part - 1.0) * c_tilt / h
    delta_y = (c_part - 1.0) * s_tilt / h
    delta_s = s_part / h

    matrix_first_part = np.array(
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

    # A rigid affine inverse is the transposed rotation followed by -R.T @ t.
    inverse_first_part = np.eye(4)
    inverse_first_part[:3, :3] = matrix_first_part[:3, :3].T
    inverse_first_part[:3, 3] = (
        -matrix_first_part[:3, :3].T @ matrix_first_part[:3, 3]
    )

    misaligned_entry = (
        matrix_first_part @ misalignment_matrix @ inverse_first_part
    )

    entry_rot_y = atan2(misaligned_entry[0, 2], misaligned_entry[2, 2])
    entry_rot_x = atan2(
        misaligned_entry[1, 2],
        hypot(misaligned_entry[1, 0], misaligned_entry[1, 1]),
    )
    entry_rot_s_no_frame = atan2(
        misaligned_entry[1, 0], misaligned_entry[1, 1]
    )

    return EntryTransform(
        shift_x=float(misaligned_entry[0, 3]),
        shift_y=float(misaligned_entry[1, 3]),
        shift_s=float(misaligned_entry[2, 3]),
        rot_y_rad=entry_rot_y,
        rot_x_rad=entry_rot_x,
        rot_s_rad_no_frame=entry_rot_s_no_frame,
    )
