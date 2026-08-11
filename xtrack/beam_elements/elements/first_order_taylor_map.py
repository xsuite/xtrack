# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

from ...base_element import BeamElement
import numpy as np
import xobjects as xo
from ...random import (
    RandomExponential,
    RandomUniformAccurate,
)
from ._common import SynchrotronRadiationRecord

class FirstOrderTaylorMap(BeamElement):
    """First order Taylor map.

    Parameters
    ----------
    length : float
        length of the element in meters.
    m0 : array_like
        6x1 array of the zero order Taylor map coefficients. Default is 0.
    m1 : array_like
        6x6 array of the first order Taylor map coefficients. Default is
        the identity matrix, so the element is an identity map by default.
    """

    isthick = True

    _xofields = {
        'length': xo.Float64,
        'm0': xo.Field(xo.Float64[6], default=np.zeros(6, dtype=np.float64)),
        'm1': xo.Field(xo.Float64[6, 6], default=np.eye(6, dtype=np.float64)),
    }

    _depends_on = [RandomUniformAccurate, RandomExponential]

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/firstordertaylormap.h"',
    ]

    _internal_record_class = SynchrotronRadiationRecord # not functional,
