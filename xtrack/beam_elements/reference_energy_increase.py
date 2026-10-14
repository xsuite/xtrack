# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

from ..base_element import BeamElement
import xobjects as xo

class ReferenceEnergyIncrease(BeamElement):

    '''Beam element modeling a change of reference energy (acceleration,
    deceleration).

    This element-driven ramp retains conventional longitudinal updates;
    the particle clock is approximate and can depend on tracking-call
    boundaries. Use the line's EnergyProgram for a coordinated clock
    and reference-energy ramp.

    Parameters
    ----------
    Delta_p0c : float
        Change in reference energy in eV. Default is ``0``.

    '''

    _xofields = {
        'Delta_p0c': xo.Float64}

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/referenceenergyincrease.h"',
    ]

    has_backtrack = True
    allow_rot_and_shift = False
