# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

from ..base_element import BeamElement
import xobjects as xo

class Wedge(BeamElement):
    """Wedge field element.

    Parameters
    ----------
    angle : float
        Angle of the wedge in radians.
    k : float
        Normalized dipole strength in units of 1/m. Default is 0.
    k1 : float
        Normalized quadrupole strength in units of 1/m^2. Default is 0.
    quad_wedge_then_dip_wedge : int
        Order of the wedge maps: 0 applies the dipole wedge followed by the
        quadrupole wedge; 1 applies the quadrupole wedge followed by the dipole
        wedge. Default is 0.
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
