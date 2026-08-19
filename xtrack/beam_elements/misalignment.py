# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

from ..base_element import BeamElement
import xobjects as xo

class Misalignment(BeamElement):
    """Beam element modeling a misalignment of a strait or curved element.

    Parameters
    ----------
    dx : float
        Misalignment in x in m.
    dy : float
        Misalignment in y in m.
    ds : float
        Misalignment in s in m.
    theta : float
        Rotation around y, yaw, positive s to x, in radians.
    phi : float
        Rotation around x, pitch, positive s to y, in radians.
    psi : float
        Rotation around s, roll, positive y to x, in radians.
    anchor : float
        Location of the misalignment as an offset in m from the element entry.
    length : float
        Length of the misaligned element in m.
    angle : float
        Angle by which the element bends the reference frame in the x-s plane.
        Direction follows the convention of the bend element, i.e. positive
        value bends x to s (opposite of phi), in radians.
    h : float
        Curvature of the element in 1/m, to be specified only for thin slices,
        i.e. when element length is zero (and therefore angle is also zero), but
        which represent slices of a curved element: in such a case curvature
        matters for the cases when ``anchor`` is not zero.
    tilt : float
        Angle (in radians) by which the element body is tilted (rolled) around
        the s-axis. Direction follows the convention of psi.
    is_exit : bool
        If False, this element brings the reference frame to the entrance of the
        misaligned element, if True, it brings the reference frame back to the
        non-misaligned frame from the exit of the element in the misaligned frame.
    """
    _xofields = {
        'dx': xo.Float64,
        'dy': xo.Float64,
        'ds': xo.Float64,
        'theta': xo.Float64,
        'phi': xo.Float64,
        'psi': xo.Float64,
        'anchor': xo.Float64,
        'length': xo.Float64,
        'angle': xo.Float64,
        'h': xo.Float64,
        'tilt': xo.Float64,
        'is_exit': xo.Int64,
    }
    has_backtrack = False
    allow_rot_and_shift = False

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/misalignment.h"',
    ]
