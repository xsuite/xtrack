# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

from ...base_element import BeamElement
import xobjects as xo
from ._common import (
    _HasIntegrator,
    _HasKnlKsl,
    _HasModelRF,
    _NOEXPR_FIELDS,
)

class TempRF(_HasKnlKsl, _HasModelRF, _HasIntegrator, BeamElement):

    isthick = True
    has_backtrack = True

    _xofields = {
        'frequency': xo.Float64,
        'voltage': xo.Float64,
        'lag': xo.Float64,
        'length': xo.Float64,
        'order': xo.Int64,
        'inv_factorial_order': xo.Float64,
        'knl': xo.Float64[:],
        'ksl': xo.Float64[:],
        'pn': xo.Float64[:],
        'ps': xo.Float64[:],
        'num_kicks': xo.Int64,
        'model': xo.Int64,
        'integrator': xo.Int64,
    }

    _rename = {
        'model': '_model',
        'integrator': '_integrator',
    }

    _noexpr_fields = _NOEXPR_FIELDS

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/temprf.h"',
    ]
