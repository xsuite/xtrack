import numpy as np

import xobjects as xo

from ..base_element import BeamElement
from ..general import _pkg_root


class LimitRect(BeamElement):
    _xofields = {
        'min_x': xo.Float64,
        'max_x': xo.Float64,
        'min_y': xo.Float64,
        'max_y': xo.Float64,
        }

LimitRect.XoStruct.extra_sources = [
        _pkg_root.joinpath('beam_elements/apertures_src/limitrect.h')]


class LimitEllipse(BeamElement):
    _xofields = {
            'a_squ': xo.Float64,
            'b_squ': xo.Float64,
            'a_b_squ': xo.Float64,
            }

    def __init__(self, a_squ=None, b_squ=None, **kwargs):
        if a_squ is None and "a" in kwargs:
            a = kwargs.get("a")
            if a is not None and a > 0.0:
                a_squ = a * a
        if a_squ is None:
            a_squ = 1.0

        if b_squ is None and "b" in kwargs:
            b = kwargs.get("b")
            if b is not None and b > 0.0:
                b_squ = b * b
        if b_squ is None:
            b_squ = 1.0

        if a_squ > 0.0 and b_squ > 0.0:
            a_b_squ = a_squ * b_squ
            kwargs['a_squ'] = a_squ
            kwargs['b_squ'] = b_squ
            kwargs['a_b_squ'] = a_squ * b_squ
            super().__init__(**kwargs)
        else:
            raise ValueError("a_squ and b_squ have to be positive definite")

    def set_half_axes(self, a, b):
        return self.set_half_axes_squ(a * a, b * b)

    def set_half_axes_squ(self, a_squ, b_squ):
        self.a_squ = a_squ
        self.b_squ = b_squ
        self.a_b_squ = a_squ * b_squ
        return self

LimitEllipse.XoStruct.extra_sources = [
        _pkg_root.joinpath('beam_elements/apertures_src/limitellipse.h')]


class LimitPolygon(BeamElement):
    _xofields = {
        'x_vertices': xo.Float64[:],
        'y_vertices': xo.Float64[:]
        }

    def __init__(self, x_vertices, y_vertices, **kwargs):

        assert len(x_vertices) == len(y_vertices)

        super().__init__(
                x_vertices=x_vertices,
                y_vertices=y_vertices,
                **kwargs)

        lengths = np.sqrt(np.diff(self.x_closed)**2
                        + np.diff(self.y_closed)**2)


    @property
    def x_closed(self):
        ctx = self._buffer.context
        xx = ctx.nparray_from_context_array(self.x_vertices)
        return np.concatenate([xx, np.array([xx[0]])])

    @property
    def y_closed(self):
        ctx = self._buffer.context
        yy = ctx.nparray_from_context_array(self.y_vertices)
        return np.concatenate([yy, np.array([yy[0]])])

LimitPolygon.XoStruct.extra_sources = [
        _pkg_root.joinpath('beam_elements/apertures_src/limitpolygon.h')]
