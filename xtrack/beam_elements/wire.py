# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

from ..base_element import BeamElement
import xobjects as xo

class Wire(BeamElement):

    '''Beam element modeling a wire (used for long range beam-beam compensation).

    Parameters
    ----------

    L_phy : float
        Physical length of the wire in meters. Default is ``0``.
    L_int : float
        Interaction length of the wire in meters. Default is ``0``.
    current : float
        Current of the wire in Ampere. Default is ``0``.
    xma : float
        Horizontal position of the wire in meters. Default is ``0``.
    yma : float
        Vertical position of the wire in meters. Default is ``0``.
    post_subtract_px : float
        Horizontal post-subtraction kick in radians. Default is ``0``.
    post_subtract_py : float
        Vertical post-subtraction kick in radians. Default is ``0``.
    '''

    _xofields={
               'L_phy'  : xo.Float64,
               'L_int'  : xo.Float64,
               'current': xo.Float64,
               'xma'    : xo.Float64,
               'yma'    : xo.Float64,

               'post_subtract_px': xo.Float64,
               'post_subtract_py': xo.Float64,
              }

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/wire.h"',
    ]
