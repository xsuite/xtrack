# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

from ..base_element import BeamElement
import xobjects as xo

class Translation(BeamElement):
    '''
    Beam element modeling a shift of the reference system, by applying
    the following transformation to the particle coordinates:

        x_new = x_old - shift_x
        y_new = y_old - shift_y

    A longitudinal shift ``shift_s`` is tracked as an exact drift of length
    ``shift_s``, compensating its advances in ``s`` and ``zeta`` so that the
    longitudinal coordinates remain consistent with ``zeta = s - beta0*c*t``
    while ``s`` stays unchanged.

    Parameters
    ----------
    shift_x : float
        Horizontal shift in meters. Default is ``0``.
    shift_y : float
        Vertical shift in meters. Default is ``0``.
    shift_s : float
        Longitudinal shift in meters. Default is ``0``.

    '''
    _xofields = {
        'shift_x': xo.Float64,
        'shift_y': xo.Float64,
        'shift_s': xo.Float64,
        }

    allow_loss_refinement = True
    has_backtrack = True
    allow_rot_and_shift = False

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/translation.h"',
    ]

    def track_frame(self, frame, backtrack=False):
        sign = -1 if backtrack else 1
        frame.translate_x(sign * self.shift_x)
        frame.translate_y(sign * self.shift_y)
        frame.translate_s(sign * self.shift_s)
