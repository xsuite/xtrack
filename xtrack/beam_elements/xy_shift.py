# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

from ..base_element import BeamElement
from ..general import DEPRECATION_INFO_PREP_1_0
from ..survey import advance_element as survey_advance_element
from warnings import warn
import xobjects as xo

class XYShift(BeamElement):
    '''
    Beam element modeling an transverse shift of the reference system, by applying
    the following transformation to the particle coordinates:

        x_new = x_old - dx
        y_new = y_old - dy

    .. warning:: The XYShift element is deprecated and will be removed in a future version. Please use the Translation element instead.

    Parameters
    ----------
    dx : float
        Horizontal shift in meters. Default is ``0``.
    dy : float
        Vertical shift in meters. Default is ``0``.

    '''
    _xofields = {
        'dx': xo.Float64,
        'dy': xo.Float64,
        }

    allow_loss_refinement = True
    has_backtrack = True
    allow_rot_and_shift = False

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/xyshift.h"',
    ]

    def __init__(self, dx=None, dy=None, **kwargs):

        warn("XYShift is deprecated and will be removed in a future version. Please use Translation instead."
             + DEPRECATION_INFO_PREP_1_0, FutureWarning, stacklevel=2)

        super().__init__(dx=dx, dy=dy, **kwargs)

    def _propagate_survey(self, v, w, backtrack):

        shift_x = self.dx
        shift_y = self.dy

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
