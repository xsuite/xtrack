# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from ..base_element import BeamElement
import xobjects as xo
from ._aperture_common import UNLIMITED


class LimitRacetrack(BeamElement):
    """
    Beam element modeling a racetrack aperture limit.

    Parameters
    ----------
    min_x : float
        Lower x limit in meters.
    max_x : float
        Upper x limit in meters.
    min_y : float
        Lower y limit in meters.
    max_y : float
        Upper y limit in meters.
    a : float
        Horizontal semi-axis in meters of ellipse used for the rounding of the corners.
    b : float
        Vertical semi-axis in meters of ellipse used for the rounding of the corners.

    """

    _xofields = {
        "min_x": xo.Field(xo.Float64, default=-UNLIMITED),
        "max_x": xo.Field(xo.Float64, default=UNLIMITED),
        "min_y": xo.Field(xo.Float64, default=-UNLIMITED),
        "max_y": xo.Field(xo.Float64, default=UNLIMITED),
        "a": xo.Float64,
        "b": xo.Float64,
    }

    has_backtrack = True

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/limitracetrack.h"'
    ]

    def __init__(
        self,
        min_x=-UNLIMITED,
        max_x=UNLIMITED,
        min_y=-UNLIMITED,
        max_y=UNLIMITED,
        a=0,
        b=0,
        **kwargs,
    ):

        if "_xobject" in kwargs:
            self.xoinitialize(_xobject=kwargs["_xobject"])
            return

        assert a >= 0
        assert b >= 0
        assert max_x >= min_x
        assert max_y >= min_y

        if a > 0.5 * (max_x - min_x) or b > 0.5 * (max_y - min_y):
            raise ValueError(
                f"Radii of corners ({a} and {b}) are larger than rectangular limit "
                f"([{min_x}, {max_x}] and [{min_y}, {max_y}])!"
            )

        super().__init__(
            min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y, a=a, b=b, **kwargs
        )
