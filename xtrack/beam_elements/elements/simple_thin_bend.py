# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

from ...base_element import BeamElement
import xobjects as xo

class SimpleThinBend(BeamElement):

    """A specialized version of Multipole to model a thin bend (ksl, hyl are all zero).

    Parameters
    ----------
    knl : array
        Normalized integrated strength of the normal components in units of m^-n.
        Must be of length 1.
    hxl : float
        Rotation angle of the reference trajectory in the horizontal plane in
        radians. Default is ``0``.
    length : float
        Length of the originating thick bend. Default is ``0``.
    """

    _xofields={
        'knl': xo.Float64[1],
        'hxl': xo.Float64,
        'length': xo.Float64,
    }

    has_backtrack = True
    allow_rot_and_shift = False

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/simplethinbend.h"',
    ]

    def __init__(self, **kwargs):

        if '_xobject' in kwargs and kwargs['_xobject'] is not None:
            self.xoinitialize(**kwargs)
            return
        knl = kwargs.get('knl')
        if knl is not None and len(knl) != 1:
            raise ValueError("For a simple thin bend, len(knl) must be 1.")

        super().__init__(**kwargs)

    @property
    def radiation_flag(self): return 0.0

    @property
    def order(self): return 0

    @property
    def inv_factorial_order(self): return 1.0
