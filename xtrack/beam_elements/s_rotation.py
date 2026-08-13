# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

from ..base_element import BeamElement
from ..general import DEPRECATION_INFO_PREP_1_0
import numpy as np
from ..survey import advance_element as survey_advance_element
from warnings import warn
import xobjects as xo
from ._common import _angle_from_trig

class SRotation(BeamElement):
    """
    Beam element modeling a rotation of the reference system around the s-axis.

    .. warning:: SRotation is deprecated and will be removed in a future version. Please use Rotation(rot_s_rad=...) instead.

    The sign convention is such that:

            px_out = px_in * cos(angle) - py_in * sin(angle)


    Parameters
    ----------
    angle : float
        Rotation angle in degrees. Default is 0.
    """

    _xofields = {
        'cos_z': xo.Float64,
        'sin_z': xo.Float64,
    }

    allow_loss_refinement = True
    has_backtrack = True
    allow_rot_and_shift = False

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/srotation.h"',
    ]

    _store_in_to_dict = ['angle']
    _skip_in_to_dict = ['sin_z', 'cos_s']

    def __init__(self, angle=None, cos_z=None, sin_z=None, **kwargs):
        """
        If either angle or a sufficient number of trig values are given,
        calculate the missing values from the others. If more than necessary
        parameters are given, their consistency will be checked.
        """

        warn("SRotation is deprecated and will be removed in a future version. "
             "Please use Rotation(rot_s_rad=...) instead. "
                + DEPRECATION_INFO_PREP_1_0,
                FutureWarning, stacklevel=2)

        if '_xobject' in kwargs and kwargs['_xobject'] is not None:
            self.xoinitialize(**kwargs)
            return

        if angle is None and (cos_z is not None or sin_z is not None):
            anglerad, cos_angle, sin_angle, _ = _angle_from_trig(cos_z, sin_z)
        elif angle is not None:
            anglerad = angle / 180 * np.pi
        else:
            anglerad = 0.0

        if cos_z is None:
            cos_z = np.cos(anglerad)
        elif not np.isclose(cos_z, np.cos(anglerad), atol=1e-13):
            raise ValueError(f'cos_z does not match angle: {cos_z} vs {anglerad}')

        if sin_z is None:
            sin_z = np.sin(anglerad)
        elif not np.isclose(sin_z, np.sin(anglerad), atol=1e-13):
            raise ValueError('sin_z does not match angle')

        super().__init__(cos_z=cos_z, sin_z=sin_z, **kwargs)

    @property
    def angle(self):
        return np.arctan2(self.sin_z, self.cos_z) * (180.0 / np.pi)

    @angle.setter
    def angle(self, value):
        anglerad = value / 180 * np.pi
        self.cos_z = np.cos(anglerad)
        self.sin_z = np.sin(anglerad)

    def _propagate_survey(self, v, w, backtrack):

        fback = 1
        if backtrack:
            fback = -1

        rx, ry, rs = 0, 0, np.deg2rad(self.angle)

        v, w = survey_advance_element(
                    v               = v,
                    w               = w,
                    length          = 0,
                    angle           = 0,
                    tilt            = 0,
                    ref_shift_x     = 0,
                    ref_shift_y     = 0,
                    ref_rot_x_rad   = fback * rx,
                    ref_rot_y_rad   = -fback * ry,
                    ref_rot_s_rad   = fback * rs,
                )

        return v, w
