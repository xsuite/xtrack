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
    _NOEXPR_FIELDS,
    _docstring_general_notes,
    _for_docstring_alignment,
    _for_docstring_edge_straight,
)

class UniformSolenoid(_HasKnlKsl, _HasIntegrator, BeamElement):

    _docstring_start = \
    """
    Uniform solenoid element with hard-edge fringe field. The axis of the
    solenoid is assumed parallel to the `s` axis. Radiation and spin
    precession are take place only in the solenoid body (no radiation and
    precession in the fringe field).

    Parameters
    ----------
    ks : float
        Strength of the solenoid component (defined as B_s / reference_rigidity)
    length : float
        Length of the element in meters.
    x0 : float, optional
        Horizontal offset of the solenoid center in meters. Defaults to 0.
    y0 : float, optional
        Vertical offset of the solenoid center in meters. Defaults to 0.
    """.strip()

    __doc__ = '\n    '.join([_docstring_start, _HasKnlKsl._for_docstring,
            _HasIntegrator._for_docstring, _for_docstring_edge_straight,
            _for_docstring_alignment, '\n', _docstring_general_notes, '\n\n'])

    isthick = True
    has_backtrack = True
    allow_loss_refinement = True

    _xofields={
        'ks': xo.Float64,
        'length': xo.Float64,
        'x0': xo.Float64,
        'y0': xo.Float64,
        'order': xo.Int64,
        'inv_factorial_order': xo.Float64,
        'knl': xo.Float64[:],
        'ksl': xo.Float64[:],
        'edge_entry_active': xo.Field(xo.UInt64, default=True),
        'edge_exit_active': xo.Field(xo.UInt64, default=True),
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
        '#include "xtrack/beam_elements/elements_src/uniform_solenoid.h"',
    ]

    def __init__(self, **kwargs):

        if 'model' in kwargs:
            raise ValueError("`model` is not supported for UniformSolenoid.")

        _HasKnlKsl.__init__(self, **kwargs)

    @property
    def _thick_slice_class(self):
        return xt.ThickSliceUniformSolenoid

    @property
    def _entry_slice_class(self):
        return xt.ThinSliceUniformSolenoidEntry

    @property
    def _exit_slice_class(self):
        return xt.ThinSliceUniformSolenoidExit
