# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

from ..base_element import BeamElement
from ..general import DEPRECATION_INFO_PREP_1_0
from warnings import warn
import xobjects as xo

class ZetaShift(BeamElement):
    '''Beam element modeling a time delay.

    .. warning:: ZetaShift is deprecated and will be removed in a future version. Please use TimeDelay instead.

    Parameters
    ----------

    dzeta : float
        Time shift dzeta in meters. Default is ``0``.

    '''

    _xofields={
        'dzeta': xo.Float64,
        }

    has_backtrack = True
    allow_rot_and_shift = False

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/zetashift.h"',
    ]

    _store_in_to_dict = ['dzeta']

    def __init__(self, *args, **kwargs):
        warn("ZetaShift is deprecated and will be removed in a future version. Please use TimeDelay instead."
             + DEPRECATION_INFO_PREP_1_0, FutureWarning, stacklevel=2)
        super().__init__(*args, **kwargs)
