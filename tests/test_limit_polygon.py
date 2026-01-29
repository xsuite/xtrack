import numpy as np

import xobjects as xo
import xtrack as xt

def _polygon_signed_area(x, y):
    x_close = np.concatenate([x, x[:1]])
    y_close = np.concatenate([y, y[:1]])
    return 0.5 * np.sum(x_close[:-1] * y_close[1:] - x_close[1:] * y_close[:-1])


def _polygon_centroid(x, y):
    x_close = np.concatenate([x, x[:1]])
    y_close = np.concatenate([y, y[:1]])
    cross = x_close[:-1] * y_close[1:] - x_close[1:] * y_close[:-1]
    area = np.sum(cross) / 2.0
    cx = np.sum((x_close[:-1] + x_close[1:]) * cross) / (6.0 * area)
    cy = np.sum((y_close[:-1] + y_close[1:]) * cross) / (6.0 * area)
    return np.array([cx, cy])


def test_limitpolygon_area_signed():
    x_ccw = np.array([-2.0, 2.0, 2.0, -2.0]) * 1e-2
    y_ccw = np.array([-1.0, -1.0, 1.5, 1.5]) * 1e-2

    aper_ccw = xt.LimitPolygon(
        x_vertices=x_ccw,
        y_vertices=y_ccw,
    )

    aper_cw = xt.LimitPolygon(
        x_vertices=x_ccw[::-1],
        y_vertices=y_ccw[::-1],
    )

    signed_expected = _polygon_signed_area(x_ccw, y_ccw)
    assert signed_expected > 0

    xo.assert_allclose(aper_ccw.area, abs(signed_expected), atol=0, rtol=0)
    xo.assert_allclose(
        aper_ccw.get_area(signed=True), signed_expected, atol=0, rtol=0
    )
    xo.assert_allclose(aper_cw.area, abs(signed_expected), atol=0, rtol=0)
    xo.assert_allclose(
        aper_cw.get_area(signed=True), -signed_expected, atol=0, rtol=0
    )


def test_limitpolygon_normals_point_inward():
    x_vertices = np.array([-1.0, 2.0, 2.5, 0.0]) * 1e-2
    y_vertices = np.array([-1.0, -1.0, 1.5, 2.0]) * 1e-2

    aper = xt.LimitPolygon(
        x_vertices=x_vertices,
        y_vertices=y_vertices,
    )

    ctx = aper._buffer.context
    Nx = ctx.nparray_from_context_array(aper.x_normal)
    Ny = ctx.nparray_from_context_array(aper.y_normal)

    xo.assert_allclose(np.sqrt(Nx**2 + Ny**2), 1.0, rtol=0, atol=1e-14)

    xv = ctx.nparray_from_context_array(aper.x_vertices)
    yv = ctx.nparray_from_context_array(aper.y_vertices)
    centroid = _polygon_centroid(xv, yv)
    xv_close = np.concatenate([xv, xv[:1]])
    yv_close = np.concatenate([yv, yv[:1]])

    for ii in range(len(xv)):
        midpoint = np.array([
            0.5 * (xv_close[ii] + xv_close[ii + 1]),
            0.5 * (yv_close[ii] + yv_close[ii + 1]),
        ])
        vec_to_center = centroid - midpoint
        normal = np.array([Nx[ii], Ny[ii]])
        assert np.dot(normal, vec_to_center) > 0


def test_limitpolygon_impact_point_and_normal():
    x_vertices = np.array([-1.0, 1.0, 1.0, -1.0]) * 1e-2
    y_vertices = np.array([-1.0, -1.0, 1.0, 1.0]) * 1e-2

    aper = xt.LimitPolygon(
        x_vertices=x_vertices,
        y_vertices=y_vertices,
    )

    ctx = aper._buffer.context
    to_ctx = ctx.nparray_to_context_array
    from_ctx = ctx.nparray_from_context_array

    x_in = to_ctx(np.array([0.0, 0.0]))
    y_in = to_ctx(np.array([0.0, 0.0]))
    z_in = to_ctx(np.array([0.0, 0.0]))
    x_out = to_ctx(np.array([2.0e-2, 0.0]))
    y_out = to_ctx(np.array([0.0, -2.0e-2]))
    z_out = to_ctx(np.array([0.0, 0.0]))

    x_int, y_int, z_int, Nx, Ny, i_found = aper.impact_point_and_normal(
        x_in=x_in,
        y_in=y_in,
        z_in=z_in,
        x_out=x_out,
        y_out=y_out,
        z_out=z_out,
    )

    x_int = from_ctx(x_int)
    y_int = from_ctx(y_int)
    z_int = from_ctx(z_int)
    Nx = from_ctx(Nx)
    Ny = from_ctx(Ny)
    i_found = from_ctx(i_found)

    xo.assert_allclose(x_int, np.array([1.0e-2, 0.0]), atol=1e-15, rtol=0)
    xo.assert_allclose(y_int, np.array([0.0, -1.0e-2]), atol=1e-15, rtol=0)
    xo.assert_allclose(z_int, np.array([0., 0]), atol=1e-15, rtol=0)
    xo.assert_allclose(Nx, np.array([-1.0, 0.0]), atol=1e-14, rtol=0)
    xo.assert_allclose(Ny, np.array([0.0, 1.0]), atol=1e-14, rtol=0)
    assert np.all(i_found >= 0)
