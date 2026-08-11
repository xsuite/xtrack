# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

from ...base_element import BeamElement
import xobjects as xo

class SimpleThinQuadrupole(BeamElement):
    """An specialized version of Multipole to model a thin quadrupole
    (knl[0], ksl, hxl, are all zero).

    Parameters
    ----------
    knl : array
        Normalized integrated strength of the normal components in units of m^-n.
        Must be of length 2.

    """

    _xofields={
        'knl': xo.Field(xo.Float64[2], default=[0, 0]),
    }

    has_backtrack = True
    allow_rot_and_shift = False

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/simplethinquadrupole.h"',
    ]

    def __init__(self, **kwargs):

        if '_xobject' in kwargs and kwargs['_xobject'] is not None:
            self.xoinitialize(**kwargs)
            return

        knl = kwargs.get('knl')
        if knl is not None and len(knl) != 2:
                raise ValueError("For a quadrupole, len(knl) must be 2.")

        super().__init__(**kwargs)

    @property
    def hxl(self): return 0.0

    @property
    def length(self): return 0.0

    @property
    def radiation_flag(self): return 0.0

    @property
    def order(self): return 1

    @property
    def inv_factorial_order(self): return 1.0
