# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

from ..base_element import BeamElement
import xobjects as xo

class TimeDelay(BeamElement):

    '''Beam element modeling a time delay, by applying the following transformation
    to the variable ``zeta``:

        zeta_new = zeta_old - shift_zeta

    Parameters
    ----------

    shift_zeta : float
        Time shift in meters added to the variable ``zeta``. Default is ``0``.

    '''

    _xofields={
        'shift_zeta': xo.Float64,
        }

    has_backtrack = True
    allow_rot_and_shift = False

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/timedelay.h"',
    ]
