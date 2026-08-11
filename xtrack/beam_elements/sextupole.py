# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

from ..base_element import BeamElement
import xobjects as xo
import xtrack as xt
from ..random import (
    RandomExponential,
    RandomUniformAccurate,
)
from ._common import (
    SynchrotronRadiationRecord,
    _HasIntegrator,
    _HasKnlKsl,
    _HasModelStraight,
    _NOEXPR_FIELDS,
    _docstring_general_notes,
    _for_docstring_alignment,
    _for_docstring_edge_straight,
)

class Sextupole(_HasKnlKsl, _HasIntegrator, _HasModelStraight, BeamElement):

    _docstring_start = \
    """
    Sextupole element.

    Parameters
    ----------
    k2 : float
        Strength of the sextupole component in m^-3.
    k2s : float
        Strength of the skew sextupole component in m^-3.
    length : float
        Length of the element in meters.
    """.strip()

    _docstring_knl_ksl_rel = \
    """knl_rel : array, optional
        Relative integrated strength of the normal components with respect to the
        main component k2 or k2s, depending on whether `main_is_skew` is False or True, respectively.
        The effect of knl_rel is added to the one of knl.
    ksl_rel : array, optional
        Relative integrated strength of the skew components with respect to the
        main component k2 or k2s, depending on whether `main_is_skew` is False or True, respectively.
        The effect of ksl_rel is added to the one of ksl.
    main_is_skew : bool, optional
        If False (default), the main component is the normal sextupole k2,
        while if True the main component is the skew sextupole k2s.
    """.strip()

    __doc__ = '\n    '.join([_docstring_start, _HasKnlKsl._for_docstring,
               _docstring_knl_ksl_rel,
               _HasModelStraight._for_docstring, _HasIntegrator._for_docstring,
               _for_docstring_edge_straight, _for_docstring_alignment, '\n',
               _docstring_general_notes, '\n\n'])

    isthick = True
    has_backtrack = True
    allow_loss_refinement = True

    _xofields={
        'k2': xo.Float64,
        'k2s': xo.Float64,
        'length': xo.Float64,
        'order': xo.Int64,
        'inv_factorial_order': xo.Float64,
        'knl': xo.Float64[:],
        'ksl': xo.Float64[:],
        'knl_rel': xo.Float64[:],
        'ksl_rel': xo.Float64[:],
        'main_is_skew': xo.Int32,
        'edge_entry_active': xo.Field(xo.UInt64, default=False),
        'edge_exit_active': xo.Field(xo.UInt64, default=False),
        'num_multipole_kicks': xo.Int64,
        'model': xo.Int64,
        'integrator': xo.Int64,
        'radiation_flag': xo.Int64,
        'delta_taper': xo.Float64,
    }

    _skip_in_to_dict = ['_order', 'inv_factorial_order']  # defined by knl, etc.

    _rename = {
        'order': '_order',
        'model': '_model',
        'integrator': '_integrator',
        'main_is_skew': '_main_is_skew',
    }

    _noexpr_fields = _NOEXPR_FIELDS

    _depends_on = [RandomUniformAccurate, RandomExponential]
    _internal_record_class = SynchrotronRadiationRecord

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/sextupole.h"',
    ]

    @property
    def main_strength(self):
        """Returns the integrated strength of the main component, i.e. k2*length
        if the main component is the normal one, or k2s*length if the main component
        is the skew one.
        """
        if self.main_is_skew:
            return self.k2s * self.length
        else:
            return self.k2 * self.length

    @property
    def _thin_slice_class(self):
        return xt.ThinSliceSextupole

    @property
    def _thick_slice_class(self):
        return xt.ThickSliceSextupole

    @property
    def _drift_slice_class(self):
        return xt.DriftSliceSextupole

    @property
    def _entry_slice_class(self):
        return xt.ThinSliceSextupoleEntry

    @property
    def _exit_slice_class(self):
        return xt.ThinSliceSextupoleExit

    @property
    def main_is_skew(self):
        """It is True if the main component is the skew one, i.e. k2s,
        or False if the main component is the normal one, i.e. k2."""
        return bool(self._main_is_skew > 0)

    @main_is_skew.setter
    def main_is_skew(self, value):
        self._main_is_skew = int(bool(value))
