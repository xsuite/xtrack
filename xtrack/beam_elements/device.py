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


class Device(_HasModelDrift, BeamElement):

    _docstring_start = """Beam element modeling a device occupying a length of
    beam line without acting on the beam, e.g. an instrument or a passive
    object.

    It tracks as a drift section, but unlike :class:`Drift` it can carry
    misalignments, so that its position and orientation are described by the
    survey. This makes it the element of choice for the objects that an
    alignment campaign measures and moves but that leave the beam undisturbed.

    Parameters
    ----------

    length : float
        Length of the device in meters. Default is ``0``.
    model : str
        Model used for the drift through the device. Available models are:
        "adaptive", "expanded", "exact". Default is "adaptive".

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

    # Unlike Drift, a device has a position in the machine: it must be able to
    # hold the misalignment that the survey gives it.
    allow_rot_and_shift = True

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/device.h"',
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
        # A device does not act on the beam, so there is nothing to thin-slice.
        return None

    @property
    def _thick_slice_class(self):
        return xt.ThickSliceDevice

    @property
    def _drift_slice_class(self):
        # Only used together with thin slices, which a device does not have.
        return None
