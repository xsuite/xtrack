# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from ..base_element import BeamElement
import xobjects as xo
from ._aperture_common import UNLIMITED


class LimitRect(BeamElement):
    """
    Beam element modeling a rectangular aperture limit.

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

    """

    _xofields = {
        "min_x": xo.Field(xo.Float64, default=-UNLIMITED),
        "max_x": xo.Field(xo.Float64, default=UNLIMITED),
        "min_y": xo.Field(xo.Float64, default=-UNLIMITED),
        "max_y": xo.Field(xo.Float64, default=UNLIMITED),
    }

    has_backtrack = True

    _extra_c_sources = ['#include "xtrack/beam_elements/elements_src/limitrect.h"']
