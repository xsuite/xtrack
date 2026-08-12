# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

from ..base_element import BeamElement
import xobjects as xo
from ._common import _HasKnlKsl

class MultipoleEdge(_HasKnlKsl, BeamElement):
    """Beam element modelling a mulipole edge.

    Parameters
    ----------
    kn: float
        Normalized integrated strength of the normal component in units of 1/m.
    ks: float
        Normalized integrated strength of the skew component in units of 1/m.
    is_exit: bool
        Flag to indicate if the edge is at the exit of the element.
    order: int
        Order of the multipole, corresponds to the length of ``kn`` and ``ks``.
    """
    _xofields = {
        'kn': xo.Float64[:],
        'ks': xo.Float64[:],
        'is_exit': xo.Int64,
        'order': xo.Int64,
    }

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/multipoleedge.h"',
    ]
    has_backtrack = True

    def __init__(self, kn: list=None, ks: list=None, is_exit=False, order=None, _xobject=None, **kwargs):
        if '_xobject' in kwargs.keys() and kwargs['_xobject'] is not None:
            self.xoinitialize(**kwargs)
            return

        multipole_kwargs = self._prepare_multipolar_params(order,
                                            skip_factorial=True, kn=kn, ks=ks)

        self.xoinitialize(is_exit=is_exit, **kwargs, **multipole_kwargs)
