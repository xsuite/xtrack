# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

from ..base_element import BeamElement
from ..general import DEPRECATION_INFO_PREP_1_0
from warnings import warn
import xobjects as xo
import xtrack as xt
from ._common import (
    _HasIntegrator,
    _HasModelRF,
    _HasModelStraight,
    _NOEXPR_FIELDS,
    _docstring_general_notes,
    _for_docstring_alignment,
)

class CrabCavity(_HasModelRF, _HasIntegrator, BeamElement):
    _docstring_start = \
    '''Crab cavity element.

    Parameters
    ----------
    length : float
        Length of the RF cavity in meters. Default is ``0``.
    crab_voltage : float
        Voltage associated to the horizontal RF deflection in Volts. Default is ``0``.
    frequency : float
        Frequency of the cavity in Hertz. It can be set only if harmonic is zero.
        Default is ``0``.
    phase : float
        Phase in radians seen at the arrival time of the reference particle (zeta = 0).
        Default is ``0``.
    lag : float
        Deprecated phase shift in degrees, added to `phase`. Default is ``0``.
    '''.strip()

    __doc__ = '\n    '.join([_docstring_start,
        _HasModelStraight._for_docstring,
        _HasIntegrator._for_docstring.replace(
            'num_multipole_kicks', 'num_kicks').replace('multipole kicks', 'kicks'),
        _for_docstring_alignment, '\n',
        _docstring_general_notes, '\n\n'])

    isthick = True
    has_backtrack = True
    allow_loss_refinement = True

    _xofields = {
        'length': xo.Float64,
        'crab_voltage': xo.Float64,
        'frequency': xo.Float64,
        'lag': xo.Float64,
        'phase': xo.Float64,
        'lag_taper': xo.Float64,
        'phase_taper': xo.Float64,
        'absolute_time': xo.Int64,
        'num_kicks': xo.Int64,
        'model': xo.Int64,
        'integrator': xo.Int64,
    }

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/crab_cavity.h"',
    ]

    _skip_in_to_dict = ['_order', 'inv_factorial_order']  # defined by knl, etc.

    _rename = {
        'model': '_model',
        'integrator': '_integrator',
        'lag': '_lag',
    }

    _noexpr_fields = _NOEXPR_FIELDS

    def __init__(self, **kwargs):

        if '_xobject' in kwargs and kwargs['_xobject'] is not None:
            self.xoinitialize(**kwargs)
            return

        model = kwargs.pop('model', None)
        integrator = kwargs.pop('integrator', None)
        lag = kwargs.pop('lag', None)

        self.xoinitialize(**kwargs)

        # Trigger properties
        if model is not None:
            self.model = model

        if integrator is not None:
            self.integrator = integrator

        if lag is not None:
            self.lag = lag

    @property
    def lag(self):
        return self._lag

    @lag.setter
    def lag(self, value):
        if value != 0:
            warn("`lag` (in degrees) is deprecated and will be removed in a future version. "
                 "Please use `phase` (in radians) instead. If you see this warning "
                 "while loading a saved line from a previous version of Xsuite, please "
                 "regenerate the line with the current version to use phase instead of lag. "
                 "Note that if both `lag` and `phase` are set, the effect is the sum of the two, "
                 " with `lag` converted to radians. "
                 + DEPRECATION_INFO_PREP_1_0,
                 FutureWarning, stacklevel=2)
        self._lag = value

    @property
    def _thin_slice_class(self):
        return xt.ThinSliceCrabCavity

    @property
    def _thick_slice_class(self):
        return xt.ThickSliceCrabCavity

    @property
    def _drift_slice_class(self):
        return xt.DriftSliceCrabCavity
