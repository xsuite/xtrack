# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from ..base_element import BeamElement
import numpy as np
import xobjects as xo
from ..svgutils import svg_to_points


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
        '#include "xtrack/beam_elements/elements_src/limitpolygon.h"'
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
            context = kwargs.get("_context", None)
            if context is None and kwargs.get("_buffer", None) is not None:
                context = kwargs["_buffer"].context
            if context is not None and isinstance(x_vertices, context.nplike_array_type):
                x_vertices = context.nparray_from_context_array(x_vertices)
            if context is not None and isinstance(y_vertices, context.nplike_array_type):
                y_vertices = context.nparray_from_context_array(y_vertices)

            if "x_normal" not in kwargs.keys():
                kwargs["x_normal"] = len(x_vertices)

            if "y_normal" not in kwargs.keys():
                kwargs["y_normal"] = len(x_vertices)

            if "resc_fac" not in kwargs.keys():
                kwargs["resc_fac"] = 1.0

            super().__init__(x_vertices=x_vertices, y_vertices=y_vertices, **kwargs)

            lengths = np.sqrt(np.diff(self.x_closed) ** 2 + np.diff(self.y_closed) ** 2)

            assert np.all(lengths > 0)

            Nx = -np.diff(self.y_closed)
            Ny = np.diff(self.x_closed)

            if self.get_area(signed=True) < 0:
                Nx = -Nx
                Ny = -Ny

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
        d = d.copy()
        for kk in (
            "min_x", "max_x", "min_y", "max_y",
            "inner_min_x", "inner_max_x", "inner_min_y", "inner_max_y",
        ):
            d.pop(kk, None)
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
        return self.get_area()

    def get_area(self, signed=False):
        out = -0.5 * np.sum(
            (self.y_closed[1:] + self.y_closed[:-1])
            * (self.x_closed[1:] - self.x_closed[:-1])
        )
        if not signed:
            out = np.abs(out)
        return out

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
