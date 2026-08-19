# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

from ..base_element import BeamElement
import xobjects as xo
import xtrack as xt
from ._common import (
    _HasModelDrift,
    _docstring_general_notes,
)

class Drift(_HasModelDrift, BeamElement):

    _docstring_start = """Beam element modeling a drift section.

    Parameters
    ----------

    length : float
        Length of the drift section in meters. Default is ``0``.
    model : str
        Model used for the drift element. Available models are: "adaptive",
        "expanded", "exact". Default is "adaptive".

    """

    __doc__ = '\n    '.join([_docstring_start, _docstring_general_notes])

    _xofields = {
        'length': xo.Float64,
        'model': xo.Int64
    }

    isthick = True
    behaves_like_drift = True
    has_backtrack = True
    allow_loss_refinement = True
    allow_rot_and_shift = False

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/drift.h"',
    ]

    _rename = {
        'model': '_model',
    }

    _noexpr_fields = {'model'}

    def __init__(self, length=None, model=None, **kwargs):

        if '_xobject' in kwargs and kwargs['_xobject'] is not None:
            self.xoinitialize(**kwargs)
            return

        if length:  # otherwise length cannot be set as a positional argument
            kwargs['length'] = length
        super().__init__(**kwargs)

        # Trigger properties
        if model is not None:
            self.model = model

    @property
    def _thin_slice_class(self):
        return None

    @property
    def _thick_slice_class(self):
        return xt.DriftSlice

    @property
    def _drift_slice_class(self):
        return xt.DriftSlice
