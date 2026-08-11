# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

from ...base_element import BeamElement
import xobjects as xo

class Wedge(BeamElement):
    """Wedge field element.

    Parameters
    ----------
    angle : float
        Angle of the wedge in radians.
    k : float
        Normalized integrated strength of the normal component in units of 1/m.
    """

    _xofields = {
        'angle': xo.Float64,
        'k': xo.Float64,
        'k1': xo.Float64,
        'quad_wedge_then_dip_wedge': xo.Int64,
    }

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/wedge.h"',
    ]
