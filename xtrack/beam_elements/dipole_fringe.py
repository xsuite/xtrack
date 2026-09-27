# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

import xobjects as xo

from ..base_element import BeamElement


class DipoleFringe(BeamElement):
    """Fringe field element of a dipole.

    Parameters
    ----------
    fint : float
        Fringe field integral in units of m^-1.
    hgap : float
        Half gap in units of m.
    k : float
        Normalized integrated strength of the normal component in units of 1/m.
    """

    _xofields = {
        'fint': xo.Float64,
        'hgap': xo.Float64,
        'k': xo.Float64,
    }

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/dipole_fringe.h"',
    ]
    has_backtrack = True

    def __init__(self, **kwargs):
        raise NotImplementedError
