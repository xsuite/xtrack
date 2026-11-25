# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

from ..base_element import BeamElement
import xobjects as xo

class ReferenceEnergyChange(BeamElement):

    '''Beam element setting the reference momentum to an absolute value.

    This element-driven ramp retains conventional longitudinal updates;
    the particle clock is approximate and can depend on tracking-call
    boundaries, and so can the phase of a cavity at a set frequency in
    `absolute_time` mode. Use the line's EnergyProgram for a coordinated
    clock and reference-energy ramp.

    Parameters
    ----------
    p0c : float
        New reference momentum in eV/c. Default is ``0``.

    '''

    _xofields = {
        'p0c': xo.Float64}

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/referenceenergychange.h"',
    ]

    allow_rot_and_shift = False
