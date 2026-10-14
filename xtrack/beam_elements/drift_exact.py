# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

from ..base_element import BeamElement
import xobjects as xo
import xtrack as xt

class DriftExact(BeamElement):
    """Beam element modeling an exact drift section.

    Parameters
    ----------

    length : float
        Length of the drift section in meters. Default is ``0``.
    """

    _xofields = {
        'length': xo.Float64
    }

    isthick = True
    behaves_like_drift = True
    has_backtrack = True
    allow_loss_refinement = True
    allow_rot_and_shift = False

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/drift_exact.h"',
    ]

    def __init__(self, length=None, **kwargs):
        if length:  # otherwise length cannot be set as a positional argument
            kwargs['length'] = length
        super().__init__(**kwargs)

    @property
    def _thin_slice_class(self):
        return None

    @property
    def _thick_slice_class(self):
        return xt.DriftExactSlice

    @property
    def _drift_slice_class(self):
        return xt.DriftExactSlice
