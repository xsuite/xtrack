# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

from ..base_element import BeamElement
from ..general import DEPRECATION_INFO_PREP_1_0
import numpy as np
from warnings import warn
import xobjects as xo
from ._common import _angle_from_trig

class YRotation(BeamElement):
    """
    Beam element modeling a rotation of the reference system around the y-axis.

    .. warning:: YRotation is deprecated and will be removed in a future version. Please use Rotation(rot_y_rad=...) instead.

    The sign convention is such that:

            px_out = px_in * cos(angle) - pz_in * sin(angle)

    Parameters
    ----------
    angle : float
        Rotation angle in degrees. Default is 0.
    """

    has_backtrack = True
    allow_loss_refinement = True
    allow_rot_and_shift = False

    _xofields={
        'sin_angle': xo.Float64,
        'cos_angle': xo.Float64,
        'tan_angle': xo.Float64,
        }

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/yrotation.h"',
    ]

    _store_in_to_dict = ['angle']
    _skip_in_to_dict = ['sin_angle', 'cos_angle', 'tan_angle']

    def __init__(
            self,
            angle=None,
            cos_angle=None,
            sin_angle=None,
            tan_angle=None,
            **kwargs,
    ):
        """
        If either angle or a sufficient number of trig values are given,
        calculate the missing values from the others. If more than necessary
        parameters are given, their consistency will be checked.
        """

        warn("YRotation is deprecated and will be removed in a future version. "
             "Please use Rotation(rot_y_rad=...) instead. "
                + DEPRECATION_INFO_PREP_1_0,
                FutureWarning, stacklevel=2)

        if '_xobject' in kwargs and kwargs['_xobject'] is not None:
            self.xoinitialize(**kwargs)
            return

        at_least_one_trig = sum(
            trig is not None for trig
                in (cos_angle, sin_angle, tan_angle)
        ) > 0

        if angle is None and at_least_one_trig:
            params = _angle_from_trig(cos_angle, sin_angle, tan_angle)
            anglerad, cos_angle, sin_angle, tan_angle = params
        elif angle is not None:
            anglerad = angle / 180 * np.pi
        else:
            anglerad = 0.0

        if cos_angle is None:
            cos_angle = np.cos(anglerad)
        elif not np.isclose(cos_angle, np.cos(anglerad), atol=1e-13):
            raise ValueError('cos_angle does not match angle')

        if sin_angle is None:
            sin_angle = np.sin(anglerad)
        elif not np.isclose(sin_angle, np.sin(anglerad), atol=1e-13, rtol=0):
            raise ValueError('sin_angle does not match angle')

        if tan_angle is None:
            tan_angle = np.tan(anglerad)
        elif not np.isclose(tan_angle, np.tan(anglerad), atol=1e-13):
            raise ValueError('tan_angle does not match angle')

        super().__init__(
            cos_angle=cos_angle, sin_angle=sin_angle, tan_angle=tan_angle,
            **kwargs)

    @property
    def angle(self):
        return np.arctan2(self.sin_angle, self.cos_angle) * (180.0 / np.pi)

    @angle.setter
    def angle(self, value):
        anglerad = value / 180 * np.pi
        self.cos_angle = np.cos(anglerad)
        self.sin_angle = np.sin(anglerad)
        self.tan_angle = np.tan(anglerad)

    def track_frame(self, frame, backtrack=False):
        sign = -1 if backtrack else 1
        frame.rotate_y(sign * np.deg2rad(self.angle))
