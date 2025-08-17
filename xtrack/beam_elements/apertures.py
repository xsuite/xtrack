# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np

import xobjects as xo

from ..base_element import BeamElement
from ..general import _pkg_root
from ..svgutils import svg_to_points


UNLIMITED = 1e10  # could use np.inf but better safe than sorry


class LimitRect(BeamElement):
    """
    Beam element modeling a rectangular aperture limit.

    Parameters
    ----------
    min_x : float
        Lower x limit in meters.
    max_x : float
        Upper x limit in meters.
    min_y : float
        Lower y limit in meters.
    max_y : float
        Upper y limit in meters.

    """

    _xofields = {
        "min_x": xo.Field(xo.Float64, default=-UNLIMITED),
        "max_x": xo.Field(xo.Float64, default=UNLIMITED),
        "min_y": xo.Field(xo.Float64, default=-UNLIMITED),
        "max_y": xo.Field(xo.Float64, default=UNLIMITED),
    }

    has_backtrack = True

    _extra_c_sources = ['#include <beam_elements/apertures_src/limitrect.h>']


class LimitRacetrack(BeamElement):
    """
    Beam element modeling a racetrack aperture limit.

    Parameters
    ----------
    min_x : float
        Lower x limit in meters.
    max_x : float
        Upper x limit in meters.
    min_y : float
        Lower y limit in meters.
    max_y : float
        Upper y limit in meters.
    a : float
        Horizontal semi-axis in meters of ellipse used for the rounding of the corners.
    b : float
        Vertical semi-axis in meters of ellipse used for the rounding of the corners.

    """

    _xofields = {
        "min_x": xo.Field(xo.Float64, default=-UNLIMITED),
        "max_x": xo.Field(xo.Float64, default=UNLIMITED),
        "min_y": xo.Field(xo.Float64, default=-UNLIMITED),
        "max_y": xo.Field(xo.Float64, default=UNLIMITED),
        "a": xo.Float64,
        "b": xo.Float64,
    }

    has_backtrack = True

    _extra_c_sources = [
        '#include <beam_elements/apertures_src/limitracetrack.h>'
    ]

    def __init__(
        self,
        min_x=-UNLIMITED,
        max_x=UNLIMITED,
        min_y=-UNLIMITED,
        max_y=UNLIMITED,
        a=0,
        b=0,
        **kwargs,
    ):

        if "_xobject" in kwargs:
            self.xoinitialize(_xobject=kwargs["_xobject"])
            return

        assert a >= 0
        assert b >= 0
        assert max_x >= min_x
        assert max_y >= min_y

        if a > 0.5 * (max_x - min_x) or b > 0.5 * (max_y - min_y):
            raise ValueError(
                f"Radii of corners ({a} and {b}) are larger than rectangular limit "
                f"([{min_x}, {max_x}] and [{min_y}, {max_y}])!"
            )

        super().__init__(
            min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y, a=a, b=b, **kwargs
        )


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
        '#include <beam_elements/apertures_src/limitellipse.h>'
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


class LimitPolygon(BeamElement):
    """
    Beam element modeling a polygonal aperture limit.

    Parameters
    ----------
    x_vertices : array_like
        x coordinates of the vertices of the polygon in meters.
    y_vertices : array_like
        y coordinates of the vertices of the polygon in meters.
    svg: dict containing
         "path"         : string describing an svg path
         "scale"       : scale from svg unit to meters default= 0.001
         "curved_steps" : steps for curved segments default=10
         "line_steps"   : steps for linear segments default=2}

    Notes
    -----
    The polygon is closed automatically by connecting the last and first vertex.

    The SVG Path follow the standard https://www.w3.org/TR/SVG/paths.html and
    can edited using https://acc-models.web.cern.ch/svg-path-editor/
    The y axis is inverted from SVG units to physical space because in svg y points downwards

    """

    _xofields = {
        "x_vertices": xo.Float64[:],
        "y_vertices": xo.Float64[:],
        "x_normal": xo.Float64[:],
        "y_normal": xo.Float64[:],
        "resc_fac": xo.Float64,
    }

    has_backtrack = True

    _extra_c_sources = [
        '#include <beam_elements/apertures_src/limitpolygon.h>'
    ]

    _kernels = {
        "LimitPolygon_impact_point_and_normal": xo.Kernel(
            c_name="LimitPolygon_impact_point_and_normal",
            args=[
                xo.Arg(xo.ThisClass, name="el"),
                xo.Arg(xo.Float64, pointer=True, name="x_in"),
                xo.Arg(xo.Float64, pointer=True, name="y_in"),
                xo.Arg(xo.Float64, pointer=True, name="z_in"),
                xo.Arg(xo.Float64, pointer=True, name="x_out"),
                xo.Arg(xo.Float64, pointer=True, name="y_out"),
                xo.Arg(xo.Float64, pointer=True, name="z_out"),
                xo.Arg(xo.Int64, pointer=False, name="n_impacts"),
                xo.Arg(xo.Float64, pointer=True, name="x_inters"),
                xo.Arg(xo.Float64, pointer=True, name="y_inters"),
                xo.Arg(xo.Float64, pointer=True, name="z_inters"),
                xo.Arg(xo.Float64, pointer=True, name="Nx_inters"),
                xo.Arg(xo.Float64, pointer=True, name="Ny_inters"),
                xo.Arg(xo.Int64, pointer=True, name="i_found"),
            ],
            n_threads="n_impacts",
        )
    }

    def __init__(self, x_vertices=None, y_vertices=None, svg=None, **kwargs):

        self.svg=svg

        if "_xobject" in kwargs.keys():
            super().__init__(**kwargs)
        else:
            if svg is not None:
                assert x_vertices is None and y_vertices is None
                path = svg["path"]
                scale = svg.get("scale", 0.001)
                curved_steps = svg.get("curved_steps", 10)
                line_steps = svg.get("line_steps", 2)
                x_vertices, y_vertices = svg_to_points(
                    path, scale=scale, curved_steps=curved_steps, line_steps=2
                )
            assert len(x_vertices) == len(y_vertices)

            if "x_normal" not in kwargs.keys():
                kwargs["x_normal"] = len(x_vertices)

            if "y_normal" not in kwargs.keys():
                kwargs["y_normal"] = len(x_vertices)

            if "resc_fac" not in kwargs.keys():
                kwargs["resc_fac"] = 1.0

            super().__init__(x_vertices=x_vertices, y_vertices=y_vertices, **kwargs)

            lengths = np.sqrt(np.diff(self.x_closed) ** 2 + np.diff(self.y_closed) ** 2)

            assert np.all(lengths > 0)

            if self.area < 0:
                self.x_vertices = self.x_vertices[::-1]
                self.y_vertices = self.y_vertices[::-1]
                # raise ValueError(
                #        "The area of the polygon is negative!\n"
                #        "Vertices must be provided with counter-clockwise order!")

            Nx = -np.diff(self.y_closed)
            Ny = np.diff(self.x_closed)

            norm_N = np.sqrt(Nx**2 + Ny**2)
            Nx = Nx / norm_N
            Ny = Ny / norm_N

            ctx = self._buffer.context
            self.x_normal = ctx.nparray_to_context_array(Nx)
            self.y_normal = ctx.nparray_to_context_array(Ny)

    def copy(self, **kwargs):
        """Copy the object."""
        out = super().copy(**kwargs)
        out.svg=None
        if self.svg is not None:
            out.svg = self.svg.copy()
        return out

    def to_dict(self, **kwargs):
        out= super().to_dict(**kwargs)
        out["svg"]=self.svg
        return out

    @classmethod
    def from_dict(cls, d, **kwargs):
        if 'svg' in d.keys() and d['svg'] is not None:
            d.pop('x_vertices')
            d.pop('y_vertices')
        out = super().from_dict(d, **kwargs)
        out.svg=d.get("svg", None)
        return out

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

    def impact_point_and_normal(self, x_in, y_in, z_in, x_out, y_out, z_out):

        ctx = self._buffer.context

        if "LimitPolygon_impact_point_and_normal" not in ctx.kernels.keys():
            # The tracking kernel requires the usual particle class
            self.compile_kernels(only_if_needed=True)

        x_inters = ctx.zeros(shape=x_in.shape, dtype=np.float64)
        y_inters = ctx.zeros(shape=x_in.shape, dtype=np.float64)
        z_inters = ctx.zeros(shape=x_in.shape, dtype=np.float64)
        Nx_inters = ctx.zeros(shape=x_in.shape, dtype=np.float64)
        Ny_inters = ctx.zeros(shape=x_in.shape, dtype=np.float64)
        i_found = ctx.zeros(shape=x_in.shape, dtype=np.int64)

        ctx.kernels.LimitPolygon_impact_point_and_normal(
            el=self,
            x_in=x_in,
            y_in=y_in,
            z_in=z_in,
            x_out=x_out,
            y_out=y_out,
            z_out=z_out,
            n_impacts=len(x_in),
            x_inters=x_inters,
            y_inters=y_inters,
            z_inters=z_inters,
            Nx_inters=Nx_inters,
            Ny_inters=Ny_inters,
            i_found=i_found,
        )

        assert np.all(i_found >= 0)

        return x_inters, y_inters, z_inters, Nx_inters, Ny_inters, i_found

    @property
    def area(self):
        return -0.5 * np.sum(
            (self.y_closed[1:] + self.y_closed[:-1])
            * (self.x_closed[1:] - self.x_closed[:-1])
        )

    @property
    def centroid(self):
        x = self.x_vertices
        y = self.x_vertices
        cx = (
            1
            / (6 * self.area)
            * np.sum((x[:-1] + x[1:]) * (x[:-1] * y[1:] - x[1:] * y[:-1]))
        )
        cy = (
            1
            / (6 * self.area)
            * np.sum((y[:-1] + y[1:]) * (y[:-1] * x[1:] - y[1:] * x[:-1]))
        )
        return (cx, cy)


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
        '#include <beam_elements/apertures_src/limitrectellipse.h>'
    ]


class LongitudinalLimitRect(BeamElement):
    """Beam element introducing a limit on the longitudinal coordinates.

    Parameters
    ----------
    min_zeta : float
        Lower limit on zeta coordinate in meters.
    max_zeta : float
        Upper limit on zeta coordinate in meters.
    min_pzeta : float
        Lower limit on pzeta coordinate.
    max_pzeta : float
        Upper limit on pzeta coordinate.
    """

    _xofields = {
        "min_zeta": xo.Field(xo.Float64, default=-UNLIMITED),
        "max_zeta": xo.Field(xo.Float64, default=UNLIMITED),
        "min_pzeta": xo.Field(xo.Float64, default=-UNLIMITED),
        "max_pzeta": xo.Field(xo.Float64, default=UNLIMITED),
    }

    has_backtrack = True

    _extra_c_sources = [
        '#include <beam_elements/apertures_src/longitudinallimitrect.h>'
    ]

    def __init__(
        self,
        min_zeta=-UNLIMITED,
        max_zeta=UNLIMITED,
        min_pzeta=-UNLIMITED,
        max_pzeta=UNLIMITED,
        **kwargs,
    ):

        super().__init__(
            min_zeta=min_zeta,
            max_zeta=max_zeta,
            min_pzeta=min_pzeta,
            max_pzeta=max_pzeta,
            **kwargs,
        )
