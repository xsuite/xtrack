# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from ..base_element import BeamElement
import xobjects as xo
from ._aperture_common import UNLIMITED


class LongitudinalLimitRect(BeamElement):
    """Beam element introducing a limit on the longitudinal coordinates.

    Parameters
    ----------
    min_zeta : float
        Lower limit on zeta coordinate in meters.
    max_zeta : float
        Upper limit on zeta coordinate in meters.
    min_pzeta : float
        Lower limit on pzeta coordinate.
    max_pzeta : float
        Upper limit on pzeta coordinate.
    """

    _xofields = {
        "min_zeta": xo.Field(xo.Float64, default=-UNLIMITED),
        "max_zeta": xo.Field(xo.Float64, default=UNLIMITED),
        "min_pzeta": xo.Field(xo.Float64, default=-UNLIMITED),
        "max_pzeta": xo.Field(xo.Float64, default=UNLIMITED),
    }

    has_backtrack = True

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/longitudinallimitrect.h"'
    ]

    def __init__(
        self,
        min_zeta=-UNLIMITED,
        max_zeta=UNLIMITED,
        min_pzeta=-UNLIMITED,
        max_pzeta=UNLIMITED,
        **kwargs,
    ):

        super().__init__(
            min_zeta=min_zeta,
            max_zeta=max_zeta,
            min_pzeta=min_pzeta,
            max_pzeta=max_pzeta,
            **kwargs,
        )
