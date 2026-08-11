# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

from ..base_element import BeamElement
import xobjects as xo

class Marker(BeamElement):
    """A marker beam element with no effect on the particles.
    """

    _xofields = {
        '_dummy': xo.Int64}

    behaves_like_drift = True
    allow_loss_refinement = True
    has_backtrack = True
    allow_rot_and_shift = False
    _skip_in_repr = ['_dummy']

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/marker.h"',
    ]
