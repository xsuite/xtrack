# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

from ..base_element import BeamElement
import xobjects as xo

class Elens(BeamElement):
    '''Beam element modeling a hollow electron lens.

    Parameters
    ----------
    inner_radius : float
        Inner radius of the electron lens in meters. Default is ``0``.
    outer_radius : float
        Outer radius of the electron lens in meters. Default is ``0``.
    current : float
        Current of the electron lens in Ampere. Default is ``0``.
    elens_length : float
        Length of the electron lens in meters. Default is ``0``.
    voltage : float
        Voltage of the electron lens in Volts. Default is ``0``.
    residual_kick_x : float
        Residual kick in the horizontal plane in radians. Default is ``0``.
    residual_kick_y : float
        Residual kick in the vertical plane in radians. Default is ``0``.
    coefficients_polynomial : array
        Array of coefficients of the polynomial. Default is ``[0]``.
    polynomial_order : int
        Order of the polynomial. Default is ``0``.

    '''

    _xofields={
        'current': xo.Float64,
        'inner_radius': xo.Field(xo.Float64, default=1.),
        'outer_radius': xo.Field(xo.Float64, default=1.),
        'elens_length': xo.Float64,
        'voltage': xo.Float64,
        'residual_kick_x': xo.Float64,
        'residual_kick_y': xo.Float64,
        'coefficients_polynomial': xo.Field(xo.Float64[:], default=[0]),
        'polynomial_order': xo.Float64,
    }

    has_backtrack = True

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/elens.h"',
    ]

    def __init__(self, **kwargs):

        if '_xobject' in kwargs and kwargs['_xobject'] is not None:
            self.xoinitialize(**kwargs)
            return

        super().__init__(**kwargs)
        polynomial_order = len(self.coefficients_polynomial) - 1
        self.polynomial_order = polynomial_order
