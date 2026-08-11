# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

from ..base_element import BeamElement
import xobjects as xo

class NonLinearLens(BeamElement):
    '''
    Beam element modeling a non-linear lens with elliptic potential.
    See the corresponding element in MAD-X documentation.

    Parameters
    ----------
    knll : float
        Integrated strength of lens (m). The strength is parametrized so that
        the quadrupole term of the multipole expansion is k1=2*knll/cnll^2.
    cnll : float
        Focusing strength (m).
        The dimensional parameter of lens (m).
        The singularities of the potential are located at x=-cnll, +cnll and y=0.
    '''

    _xofields={
        'knll': xo.Float64,
        'cnll': xo.Float64,
    }

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/nonlinearlens.h"',
    ]
