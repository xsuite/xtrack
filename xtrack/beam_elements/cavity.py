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

class Cavity(_HasModelRF, _HasIntegrator, BeamElement):

    _docstring_start = \
    '''RF cavity element.

    Parameters
    ----------
    length : float
        Length of the RF cavity in meters. Default is ``0``.
    voltage : float
        Voltage of the RF cavity in Volts. Default is ``0``.
    frequency : float
        Frequency of the RF cavity in Hertz. It can be set only if harmonic is zero.
        Default is ``0``.
    harmonic : float
        Harmonic number of the RF cavity. It can be set only if frequency is zero.
        If `harmonic` is non-zero, the frequency is computed from the length of the
        beam_line and the speed of the reference particle (beta0 * clight).
        When `harmonic` is set, the cavity can only be used within a Line and not
        in standalone tracking (i.e. Cavity.track(...) will raise an error).
        Default is ``0``.
    phase : float
        Phase in radians seen at the arrival time of the reference particle (zeta = 0).
        When `absolute_time` is True, `phase` is the phase at time zero. Default is ``0``.
    lag : float
        Deprecated phase shift in degrees, added to `phase`. Default is ``0``.
    absolute_time : bool
        If True, the cavity phase is computed from the absolute time of the
        simulation, otherwise the cavity is synchronized with the arrival time of
        the reference particle (zeta=0). Default is False.
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
        'voltage': xo.Float64,
        'frequency': xo.Float64,
        'lag': xo.Float64,
        'phase': xo.Float64,
        'harmonic': xo.Float64,
        'lag_taper': xo.Float64,
        'phase_taper': xo.Float64,
        'absolute_time': xo.Int64,
        'num_kicks': xo.Int64,
        'model': xo.Int64,
        'integrator': xo.Int64,
    }

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/cavity.h"',
    ]

    _skip_in_to_dict = ['_order', 'inv_factorial_order']  # defined by knl, etc.

    _rename = {
        'model': '_model',
        'integrator': '_integrator',
        'frequency': '_frequency',
        'harmonic': '_harmonic',
        'lag': '_lag',
    }

    _default_frequency = 0.0
    _default_harmonic = 0.0

    _noexpr_fields = _NOEXPR_FIELDS

    def __init__(self, **kwargs):

        if '_xobject' in kwargs and kwargs['_xobject'] is not None:
            self.xoinitialize(**kwargs)
            return

        model = kwargs.pop('model', None)
        integrator = kwargs.pop('integrator', None)
        frequency = kwargs.pop('frequency', None)
        harmonic = kwargs.pop('harmonic', None)
        lag = kwargs.pop('lag', None)

        self.xoinitialize(**kwargs)

        # Trigger properties
        if model is not None:
            self.model = model

        if integrator is not None:
            self.integrator = integrator

        if frequency is not None:
            self.frequency = frequency

        if harmonic is not None:
            self.harmonic = harmonic

        if lag is not None:
            self.lag = lag

    def track(self, particles, *args, **kwargs):

        if self.harmonic != 0:
            raise RuntimeError("Cavity cannot be used in standalone tracking "
                               "when harmonic is not zero. Please use the "
                               "cavity within a Line or set frequency instead"
                               " of harmonic.")
        return super().track(particles, *args, **kwargs)

    @property
    def frequency(self):
        return self._frequency

    @frequency.setter
    def frequency(self, value):
        if self._harmonic != 0 and value != 0:
            raise ValueError("Cannot set non-zero frequency when harmonic is not zero.")
        self._frequency = value

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
    def harmonic(self):
        return self._harmonic

    @harmonic.setter
    def harmonic(self, value):
        if self._frequency != 0 and value != 0:
            raise ValueError("Cannot set non-zero harmonic when frequency is not zero.")
        self._harmonic = value

    @property
    def _thin_slice_class(self):
        return xt.ThinSliceCavity

    @property
    def _thick_slice_class(self):
        return xt.ThickSliceCavity

    @property
    def _drift_slice_class(self):
        return xt.DriftSliceCavity
