# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

from ..base_element import BeamElement
from ..survey import advance_element as survey_advance_element
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

    def _propagate_survey(self, v, w, backtrack):

        shift_x = self.shift_x
        shift_y = self.shift_y

        if backtrack:
            fback = -1
        else:
            fback = 1

        v, w = survey_advance_element(
                    v               = v,
                    w               = w,
                    length          = 0,
                    angle           = 0,
                    tilt            = 0,
                    ref_shift_x     = fback * shift_x,
                    ref_shift_y     = fback * shift_y,
                    ref_rot_x_rad   = 0,
                    ref_rot_y_rad   = 0,
                    ref_rot_s_rad   = 0,
                )
        return v, w
