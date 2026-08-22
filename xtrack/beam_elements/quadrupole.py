# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

from ..base_element import BeamElement, FloatOrTpsa
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

class Quadrupole(_HasKnlKsl, _HasIntegrator, _HasModelStraight, BeamElement):

    _docstring_start = \
    """
    Quadrupole element.

    Parameters
    ----------
    k1 : float
        Strength of the quadrupole component in m^-2.
    k1s : float
        Strength of the skew quadrupole component in m^-2.
    length : float
        Length of the element in meters.
    """.strip()

    _docstring_knl_rel_ksl_rel = \
    """
    knl_rel : array, optional
        Relative integrated strength of the normal components with respect to the
        main component k1 or k1s, depending whether `main_is_skew` is False or True, respectively.
        The effect of knl_rel is added to the one of knl.
    ksl_rel : array, optional
        Relative integrated strength of the skew components with respect to the
        main component k1 or k1s, depending whether `main_is_skew` is False or True, respectively.
        The effect of ksl_rel is added to the one of ksl.
    main_is_skew : bool, optional
        If True, the main component is the skew one (k1s), otherwise it is the normal one (k1).
        Default is False.
    """.strip()

    __doc__ = '\n    '.join([_docstring_start, _HasKnlKsl._for_docstring,
               _docstring_knl_rel_ksl_rel,
               _HasModelStraight._for_docstring, _HasIntegrator._for_docstring,
               _for_docstring_edge_straight, _for_docstring_alignment, '\n',
               _docstring_general_notes, '\n\n'])

    isthick = True
    has_backtrack = True
    allow_loss_refinement = True

    _xofields = {
        'k1': FloatOrTpsa,
        'k1s': FloatOrTpsa,
        'length': xo.Float64,
        'num_multipole_kicks': xo.Int64,
        'order': xo.Int64,
        'inv_factorial_order': xo.Float64,
        'knl': xo.Float64[:],
        'ksl': xo.Float64[:],
        'knl_rel': xo.Float64[:],
        'ksl_rel': xo.Float64[:],
        'main_is_skew': xo.Int32,
        'edge_entry_active': xo.Field(xo.UInt64, default=False),
        'edge_exit_active': xo.Field(xo.UInt64, default=False),
        'model': xo.Int64,
        'integrator': xo.Int64,
        'radiation_flag': xo.Int64,
        'delta_taper': xo.Float64,
        '_tpsa_enabled': xo.Field(xo.Int8, default=0),
    }

    _skip_in_to_dict = [
        '_order', 'inv_factorial_order', '_tpsa_enabled']  # defined by knl, etc.

    _rename = {
        'order': '_order',
        'model': '_model',
        'integrator': '_integrator',
        'main_is_skew': '_main_is_skew',
    }

    _noexpr_fields = _NOEXPR_FIELDS

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/quadrupole.h"',
    ]

    _depends_on = [RandomUniformAccurate, RandomExponential]

    _internal_record_class = SynchrotronRadiationRecord

    @property
    def main_strength(self):
        """Returns the integrated strength of the main component, i.e. k1*length
        if the main component is the normal one, or k1s*length if the main component
        is the skew one.
        """
        if self.main_is_skew:
            return self.k1s * self.length
        else:
            return self.k1 * self.length

    @property
    def radiation_flag(self): return 0.0

    @property
    def _thin_slice_class(self):
        return xt.ThinSliceQuadrupole

    @property
    def _thick_slice_class(self):
        return xt.ThickSliceQuadrupole

    @property
    def _drift_slice_class(self):
        return xt.DriftSliceQuadrupole

    @property
    def _entry_slice_class(self):
        return xt.ThinSliceQuadrupoleEntry

    @property
    def _exit_slice_class(self):
        return xt.ThinSliceQuadrupoleExit

    @property
    def main_is_skew(self):
        """It is True if the main component is the skew one, i.e. k1s,
        or False if the main component is the normal one, i.e. k1."""
        return bool(self._main_is_skew > 0)

    @main_is_skew.setter
    def main_is_skew(self, value):
        self._main_is_skew = int(bool(value))
