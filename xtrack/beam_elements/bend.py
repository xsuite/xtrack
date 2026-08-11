# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

from ..base_element import BeamElement
import xtrack as xt
from ..random import (
    RandomExponential,
    RandomUniformAccurate,
)
from ._common import (
    SynchrotronRadiationRecord,
    _BendCommon,
    _HasIntegrator,
    _HasKnlKsl,
    _HasModelCurved,
    _NOEXPR_FIELDS,
    _docstring_general_notes,
    _for_docstring_alignment,
    _for_docstring_edge_bend,
)

class Bend(_BendCommon, BeamElement):

    _docstring_start = \
    """Bending magnet element, sector-bend type.

    Parameters
    ----------
    length : float
        Length of the element in meters along the reference trajectory.
    angle : float
        Angle of the bend in radians. This is the angle by which the reference
        trajectory is bent in the horizontal plane.
    k0 : float, optional
        Strength of the horizontal dipolar component in units of m^-1.
        It can be set to the string value 'from_h', in which case `k0` is
        computed from the curvature defined by `angle` and `length`
        (i.e. `k0 = h = angle/length`) and `k0_from_h` is set to True.
    k1 : float, optional
        Strength of the quadrupolar component in units of m^-2.
    k2 : float, optional
        Strength of the sextupolar component in units of m^-3.
    k0_from_h : bool, optional
        If True, `k0` is computed from the curvature defined by `angle` and
        `length` (i.e. `k0 = h = angle/length`). Default is True. The flag
        becomes false when `k0` is set directly to a numeric value.
    """.strip()

    _docstring_knl_rel_ksl_rel = \
    """knl_rel : array, optional
        Relative integrated strength of the normal components with respect to the
        main component k0. The effect of knl_rel is added to the one of knl.
    ksl_rel : array, optional
        Relative integrated strength of the skew components with respect to the
        main component k0. The effect of ksl_rel is added to the one of ksl.
    """.strip()

    __doc__ = '\n    '.join([_docstring_start, _HasKnlKsl._for_docstring,
            _docstring_knl_rel_ksl_rel,
            _HasModelCurved._for_docstring, _HasIntegrator._for_docstring,
            _for_docstring_edge_bend, _for_docstring_alignment, '\n',
            _docstring_general_notes, '\n\n'])

    allow_loss_refinement = True

    _xofields = _BendCommon._common_xofields
    _rename = _BendCommon._common_rename

    _depends_on = [RandomUniformAccurate, RandomExponential]

    _internal_record_class = SynchrotronRadiationRecord

    _noexpr_fields = _NOEXPR_FIELDS

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/bend.h"',
    ]

    def __init__(self, **kwargs):

        if '_xobject' in kwargs and kwargs['_xobject'] is not None:
            self.xoinitialize(**kwargs)
            return

        if 'h' in kwargs:
            raise ValueError("Setting `h` directly is not allowed. "
                                "Set `length` and `angle` instead.")

        to_be_set_with_properties = []
        for nn in ['length', 'angle', 'k0_from_h', 'edge_entry_model',
                   'edge_exit_model', 'k0']:
            if nn in kwargs:
                to_be_set_with_properties.append((nn, kwargs.pop(nn)))

        _HasKnlKsl.__init__(self, **kwargs)

        for nn, val in to_be_set_with_properties:
            setattr(self, nn, val)

    @property
    def _thin_slice_class(self):
        return xt.ThinSliceBend

    @property
    def _thick_slice_class(self):
        return xt.ThickSliceBend

    @property
    def _drift_slice_class(self):
        return xt.DriftSliceBend

    @property
    def _entry_slice_class(self):
        return xt.ThinSliceBendEntry

    @property
    def _exit_slice_class(self):
        return xt.ThinSliceBendExit

    @property
    def _repr_fields(self):
        return super()._repr_fields
