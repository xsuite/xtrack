# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

from ...base_element import BeamElement
import xobjects as xo
import xtrack as xt
from ...random import (
    RandomExponential,
    RandomUniformAccurate,
)
from ._common import (
    SynchrotronRadiationRecord,
    _HasIntegrator,
    _HasKnlKsl,
    _HasModelCurved,
    _HasModelStraight,
    _NOEXPR_FIELDS,
    _docstring_general_notes,
    _for_docstring_alignment,
    _for_docstring_edge_straight,
    _handle_knl_ksl_rel_kwargs,
)

class Multipole(_HasKnlKsl, _HasModelStraight, _HasIntegrator, BeamElement):

    _docstring_start = \
    """Beam element modeling a magnetic multipole.

    Parameters
    ----------

    knl : array
        Integrated strength of the normal components in units of m^-n.
    ksl : array
        Integrated strength of the skew components in units of m^-n.
    order : int
        Order of the multipole. By default it is inferred from the length of
        knl and ksl.
    hxl : float
        Rotation angle in radians applied to the reference trajectory in the
        horizontal plane. Default is ``0``.
    length : float
        Length of the originating thick multipole. Default is ``0``.
    isthick : bool
        Whether the multipole is to be treated as thick (True) or thin (False).
        Default is ``False``.
    """.strip()

    _docstring_knl_rel_ksl_rel = \
    """
    knl_rel : array
        Relative integrated strength of the normal components with respect to the main
        component defined by `main_order` and `main_is_skew`. The effect of
        `knl_rel` is added to the one of `knl`.
    ksl_rel : array
        Relative integrated strength of the skew components with respect to the main
        component defined by `main_order` and `main_is_skew`. The effect of
        `ksl_rel` is added to the one of `ksl`.
    main_order : int
        Order of the main multipole component used for defining the relative strengths
        `knl_rel` and `ksl_rel`. Default is ``0``.
    main_is_skew : bool
        Whether the main multipole component used for defining the relative strengths
        `knl_rel` and `ksl_rel` is skew (True) or normal (False). Default is ``False``.
    """.strip()

    __doc__ = '\n    '.join([_docstring_start, _docstring_knl_rel_ksl_rel,
                             _HasModelCurved._for_docstring,
                             _HasIntegrator._for_docstring,
                             _for_docstring_edge_straight,
                             _for_docstring_alignment, '\n',
                             _docstring_general_notes, '\n\n'])

    #isthick can be changed dynamically for this element

    has_backtrack = True

    _xofields={
        'order': xo.Int64,
        'inv_factorial_order': xo.Float64,
        'length': xo.Float64,
        'hxl': xo.Float64,
        'radiation_flag': xo.Int64,
        'delta_taper': xo.Float64,
        'knl': xo.Float64[:],
        'ksl': xo.Float64[:],
        'knl_rel': xo.Float64[:],
        'ksl_rel': xo.Float64[:],
        'main_order': xo.Int32,
        'main_is_skew': xo.Int32,
        'isthick': xo.Int64,
        'num_multipole_kicks': xo.Int64,
        'model': xo.Int64,
        'integrator': xo.Int64,
        }

    _rename = {
        'order': '_order',
        'isthick': '_isthick',
        'model': '_model',
        'integrator': '_integrator',
        'main_is_skew': '_main_is_skew',
    }

    _noexpr_fields = _NOEXPR_FIELDS

    _skip_in_to_dict = ['_order', 'inv_factorial_order']  # defined by knl, etc.

    _depends_on = [RandomUniformAccurate, RandomExponential]

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/multipole.h"',
    ]

    _internal_record_class = SynchrotronRadiationRecord

    @property
    def allow_loss_refinement(self):
        '''
        Loss refinement is allowed only for thick multipoles with non-zero length.
        '''
        # Allow refinement only when thick (to keep old behavior when thin and
        # have consistency with other thick elements otherwise)
        return self.isthick and self.length != 0

    @property
    def main_strength(self):
        if self.main_is_skew:
            return self.ksl[self.main_order]
        else:
            return self.knl[self.main_order]

    def __init__(self, **kwargs):

        if '_xobject' in kwargs.keys() and kwargs['_xobject'] is not None:
            self.xoinitialize(**kwargs)
            return

        order = kwargs.pop('order', None)
        knl = kwargs.pop('knl', None)
        ksl = kwargs.pop('ksl', None)

        multipolar_kwargs = self._prepare_multipolar_params(order, knl=knl, ksl=ksl)
        kwargs.update(multipolar_kwargs)

        model = kwargs.pop('model', None)
        integrator = kwargs.pop('integrator', None)
        isthick = kwargs.pop('isthick', None)

        if "bal" in kwargs.keys():
            raise ValueError("`bal` not supported anymore")

        if 'hyl' in kwargs.keys():
            assert kwargs['hyl'] == 0.0, 'hyl is not supported anymore'

        _handle_knl_ksl_rel_kwargs(kwargs)

        self.xoinitialize(**kwargs)

        # Trigger properties
        if model is not None:
            self.model = model

        if integrator is not None:
            self.integrator = integrator

        if isthick is not None:
            self.isthick = isthick

    @property
    def hyl(self):
        raise ValueError("hyl is not anymore supported")

    @hyl.setter
    def hyl(self, value):
        raise ValueError("hyl is not anymore supported")

    @property
    def isthick(self):
        return bool(self._isthick > 0)

    @isthick.setter
    def isthick(self, value):
        self._isthick = int(bool(value))

    @property
    def _thin_slice_class(self):
        return xt.ThinSliceMultipole

    @property
    def _thick_slice_class(self):
        return xt.ThickSliceMultipole

    @property
    def _drift_slice_class(self):
        return xt.DriftSliceMultipole

    @property
    def main_is_skew(self):
        return bool(self._main_is_skew > 0)

    @main_is_skew.setter
    def main_is_skew(self, value):
        self._main_is_skew = int(bool(value))
