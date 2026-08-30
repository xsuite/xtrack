# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

from ..base_element import BeamElement
import xobjects as xo

class Translation(BeamElement):
    '''
    Beam element modeling a transverse shift of the reference system, by applying
    the following transformation to the particle coordinates:

        x_new = x_old - shift_x
        y_new = y_old - shift_y

    Parameters
    ----------
    shift_x : float
        Horizontal shift in meters. Default is ``0``.
    shift_y : float
        Vertical shift in meters. Default is ``0``.

    '''
    _xofields = {
        'shift_x': xo.Float64,
        'shift_y': xo.Float64,
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
