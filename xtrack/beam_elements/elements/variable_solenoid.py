# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

from ...base_element import BeamElement
import xobjects as xo
from ...random import (
    RandomExponential,
    RandomUniformAccurate,
)
from ._common import (
    SynchrotronRadiationRecord,
    _HasIntegrator,
    _HasKnlKsl,
    _NOEXPR_FIELDS,
    _docstring_general_notes,
    _for_docstring_alignment,
)

class VariableSolenoid(_HasKnlKsl, _HasIntegrator, BeamElement):

    _docstring_start = \
    """
    Solenoid with linearly varying lingitudinal field. The transverse fields
    arising form the derivative of the longitudinal fields are taken into account
    in particle dynamics, radiation, spin precession.

    Parameters
    ----------
    ks_profile : array-like of 2 floats
        Solenoid strength at entry and exit of the element (defined as
        B_s / reference_rigidity).
    length : float
        Length of the element in meters along the reference trajectory.
    x0 : float, optional
        Horizontal offset of the solenoid center in meters. Defaults to 0.
    y0 : float, optional
        Vertical offset of the solenoid center in meters. Defaults to 0.
    """.strip()

    __doc__ = '\n    '.join([_docstring_start, _HasKnlKsl._for_docstring,
        _HasIntegrator._for_docstring, _for_docstring_alignment, '\n',
        _docstring_general_notes, '\n\n'])

    isthick = True
    has_backtrack = True

    _xofields={
        'ks_profile': xo.Float64[2],
        'length': xo.Float64,
        'x0': xo.Float64,
        'y0': xo.Float64,
        'order': xo.Int64,
        'inv_factorial_order': xo.Float64,
        'knl': xo.Float64[:],
        'ksl': xo.Float64[:],
        'num_multipole_kicks': xo.Int64,
        'integrator': xo.Int64,
        'radiation_flag': xo.Int64,
        'delta_taper': xo.Float64,
    }

    _skip_in_to_dict = ['_order', 'inv_factorial_order']  # defined by knl, etc.

    _rename = {
        'order': '_order',
        'integrator': '_integrator',
    }

    _noexpr_fields = _NOEXPR_FIELDS

    _depends_on = [RandomUniformAccurate, RandomExponential]
    _internal_record_class = SynchrotronRadiationRecord

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/variable_solenoid.h"',
    ]

    def __init__(self, **kwargs):

        if 'model' in kwargs:
            raise ValueError("`model` is not supported for UniformSolenoid.")

        _HasKnlKsl.__init__(self, **kwargs)
