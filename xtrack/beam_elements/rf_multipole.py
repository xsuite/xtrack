# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

from ..base_element import BeamElement
from ..general import DEPRECATION_INFO_PREP_1_0
from warnings import warn
import xobjects as xo
from ._common import (
    _HasKnlKsl,
    _docstring_general_notes,
    _for_docstring_alignment,
)

class RFMultipole(_HasKnlKsl, BeamElement):

    _docstring_start = \
    """Beam element modeling a thin modulated multipole, with strengths
    dependent on the z coordinate:

    Parameters
    ----------
    frequency : float
        Frequency in Hertz. Default is ``0``.
    knl : array
        Integrated strength of the normal rf-multipole components in units of m^-n.
    ksl : array
        Integrated strength of the skew rf-multipole components in units of m^-n.
    order : int
        Order of the multipole. If not provided, it will be inferred from knl and/or ksl.
    phase_n : array
        Phase of the normal components in radians.
    phase_s : array
        Phase of the skew components in radians.
    pn : array
        Deprecated. Phase of the normal components in degrees.
    ps : array
        Deprecated. Phase of the skew components in degrees.
    voltage : float
        Longitudinal voltage. Default is ``0``.
    phase : float
        Longitudinal phase in radians seen by the reference particle. Default is ``0``.
    lag : float
        Deprecated longitudinal phase in degrees, added to `phase`. Default is ``0``.
    """.strip()

    __doc__ = '\n    '.join([_docstring_start, _for_docstring_alignment, '\n',
                             _docstring_general_notes, '\n\n'])

    _xofields={
        'voltage': xo.Float64,
        'frequency': xo.Float64,
        'lag': xo.Float64,
        'phase': xo.Float64,
        'order': xo.Int64,
        'inv_factorial_order': xo.Float64,
        'knl': xo.Float64[:],
        'ksl': xo.Float64[:],
        'pn': xo.Float64[:],
        'ps': xo.Float64[:],
        'phase_n': xo.Float64[:],
        'phase_s': xo.Float64[:],
        'absolute_time': xo.Int64,
    }

    has_backtrack = True
    allow_loss_refinement = True

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/rfmultipole.h"',
    ]

    _skip_in_to_dict = ['_order', 'inv_factorial_order']  # defined by knl, etc.

    _rename = {
        'order': '_order',
        'lag': '_lag',
        'pn': '_pn',
        'ps': '_ps',
    }

    def __init__(self, **kwargs):

        if '_xobject' in kwargs and kwargs['_xobject'] is not None:
            self.xoinitialize(**kwargs)
            return

        pn = kwargs.get('pn')
        ps = kwargs.get('ps')
        lag = kwargs.pop('lag', None)
        if pn is not None:
            self._warn_if_deprecated_phase_is_nonzero(pn, 'pn', 'phase_n')
        if ps is not None:
            self._warn_if_deprecated_phase_is_nonzero(ps, 'ps', 'phase_s')

        super().__init__(**kwargs)

        if lag is not None:
            self.lag = lag

    @property
    def lag(self):
        return self._lag

    @lag.setter
    def lag(self, value):
        if value != 0:
            warn("`lag` (in degrees) is deprecated and will be removed in a future version. "
                 "Please use `phase` (in radians) instead. "
                 "Note that if both `lag` and `phase` are set, the effect is the sum of the two,"
                 " with `lag` converted to radians. "
                 + DEPRECATION_INFO_PREP_1_0,
                 FutureWarning, stacklevel=2)
        self._lag = value

    @property
    def pn(self):
        return self._buffer.context.linked_array_type.from_array(
            self._pn,
            mode='setitem_from_container',
            container=self,
            container_setitem_name='_pn_setitem')

    @pn.setter
    def pn(self, value):
        self.pn[:] = value

    def _pn_setitem(self, index, value):
        self._warn_if_deprecated_phase_is_nonzero(value, 'pn', 'phase_n')
        self._pn[index] = value

    @property
    def ps(self):
        return self._buffer.context.linked_array_type.from_array(
            self._ps,
            mode='setitem_from_container',
            container=self,
            container_setitem_name='_ps_setitem')

    @ps.setter
    def ps(self, value):
        self.ps[:] = value

    def _ps_setitem(self, index, value):
        self._warn_if_deprecated_phase_is_nonzero(value, 'ps', 'phase_s')
        self._ps[index] = value
