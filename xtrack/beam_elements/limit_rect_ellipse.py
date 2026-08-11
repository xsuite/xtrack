# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from ..base_element import BeamElement
import numpy as np
import xobjects as xo
from ._aperture_common import UNLIMITED


class LimitRectEllipse(BeamElement):
    """
    Element modeling an aperture limit given by the intersection of
    a symmetric LimitRect and a LimitEllipse.

    The particles are lost if they exceed either the rect or ellipse aperture.

    Parameters
    ----------
    max_x : float
        Horizontal semi-axis of rect in meters.
    max_y : float
        Vertical semi-axis of rect in meters.
    a : float
        Horizontal semi-axis of ellipse in meters.
    b : float
        Vertical semi-axis of ellipse in meters.
    """

    _xofields = {
        "max_x": xo.Field(xo.Float64, default=UNLIMITED),
        "max_y": xo.Field(xo.Float64, default=UNLIMITED),
        "a_squ": xo.Float64,
        "b_squ": xo.Float64,
        "a_b_squ": xo.Float64,
    }

    has_backtrack = True

    def __init__(
        self,
        max_x=UNLIMITED,
        max_y=UNLIMITED,
        a_squ=None,
        b_squ=None,
        a=None,
        b=None,
        **kwargs,
    ):

        if a is None and a_squ is None:
            a = UNLIMITED

        if b is None and b_squ is None:
            b = UNLIMITED

        if a is not None:
            a_squ = a * a

        if b is not None:
            b_squ = b * b

        if max_x < 0.0:
            raise ValueError("max_x has to be positive definite")

        if max_y < 0.0:
            raise ValueError("max_y has to be_positive definite")

        if a_squ < 0.0 or b_squ < 0.0:
            raise ValueError("a_squ and b_squ have to be positive definite")

        if "a_b_squ" not in kwargs.keys():
            kwargs["a_b_squ"] = a_squ * b_squ

        super().__init__(max_x=max_x, max_y=max_y, a_squ=a_squ, b_squ=b_squ, **kwargs)

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
        return ["max_x", "max_y", "a", "b"]

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/limitrectellipse.h"'
    ]
