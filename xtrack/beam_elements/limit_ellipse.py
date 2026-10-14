# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from ..base_element import BeamElement
import numpy as np
import xobjects as xo
from ._aperture_common import UNLIMITED


class LimitEllipse(BeamElement):
    """
    Beam element modeling an elliptical aperture limit.

    Parameters
    ----------
    a : float
        Horizontal semi-axis in meters.
    b : float
        Vertical semi-axis in meters.

    """

    _xofields = {
        "a_squ": xo.Float64,
        "b_squ": xo.Float64,
        "a_b_squ": xo.Float64,
    }

    has_backtrack = True

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/limitellipse.h"'
    ]

    def to_dict(self):
        dct = super().to_dict()
        dct["a"] = np.sqrt(self.a_squ)
        dct["b"] = np.sqrt(self.b_squ)
        return dct

    def __init__(self, a=None, b=None, a_squ=None, b_squ=None, **kwargs):

        if a is None and a_squ is None:
            a = UNLIMITED

        if b is None and b_squ is None:
            b = UNLIMITED

        if a is not None:
            a_squ = a * a

        if b is not None:
            b_squ = b * b

        if "a_b_squ" not in kwargs.keys():
            kwargs["a_b_squ"] = a_squ * b_squ

        if a_squ > 0.0 and b_squ > 0.0:
            super().__init__(a_squ=a_squ, b_squ=b_squ, **kwargs)
        else:
            raise ValueError("a_squ and b_squ have to be positive definite")

    @property
    def a(self):
        return np.sqrt(self.a_squ)

    @a.setter
    def a(self, a):
        self.a_squ = a * a
        self.a_b_squ = self.a_squ * self.b_squ

    @property
    def b(self):
        return np.sqrt(self.b_squ)

    @b.setter
    def b(self, b):
        self.b_squ = b * b
        self.a_b_squ = self.a_squ * self.b_squ

    def set_half_axes(self, a, b):
        return self.set_half_axes_squ(a * a, b * b)

    def set_half_axes_squ(self, a_squ, b_squ):
        self.a_squ = a_squ
        self.b_squ = b_squ
        self.a_b_squ = a_squ * b_squ
        return self

    @property
    def _repr_fields(self):
        return ["a", "b"]
