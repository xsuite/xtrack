import numpy as np

import xobjects as xo
import xtrack as xt
import xpart as xp

def test_rect_ellipse():

    for context in xo.context.get_test_contexts():
        print(f"Test {context.__class__}")

        np2ctx = context.nparray_to_context_array
        ctx2np = context.nparray_from_context_array

        aper_rect_ellipse = xt.LimitRectEllipse(_context=context,
                max_x=23e-3, max_y=18e-3, a=23e-2, b=23e-2)
        aper_ellipse = xt.LimitRectEllipse(_context=context,
                                           a=23e-2, b=23e-2)
        aper_rect = xt.LimitRect(_context=context,
                                 max_x=23e-3, min_x=-23e-3,
                                 max_y=18e-3, min_y=-18e-3)

        XX, YY = np.meshgrid(np.linspace(-30e-3, 30e-3, 100),
                             np.linspace(-30e-3, 30e-3, 100))
        x_part = XX.flatten()
        y_part = XX.flatten()
        part_re = xp.Particles(_context=context,
                               x=x_part, y=y_part)
        part_e = part_re.copy()
        part_r = part_re.copy()

        aper_rect_ellipse.track(part_re)
        aper_ellipse.track(part_e)
        aper_rect.track(part_r)

        flag_re = ctx2np(part_re.state)[np.argsort(ctx2np(part_re.particle_id))]
        flag_r = ctx2np(part_r.state)[np.argsort(ctx2np(part_r.particle_id))]
        flag_e = ctx2np(part_e.state)[np.argsort(ctx2np(part_e.particle_id))]

        assert np.all(flag_re == (flag_r & flag_e))

def test_aperture_polygon():

    for context in xo.context.get_test_contexts():
        print(f"Test {context.__class__}")

        np2ctx = context.nparray_to_context_array
        ctx2np = context.nparray_from_context_array

        x_vertices=np.array([1.5, 0.2, -1, -1,  1])*1e-2
        y_vertices=np.array([1.3, 0.5,  1, -1, -1])*1e-2

        aper = xt.LimitPolygon(
                        _context=context,
                        x_vertices=np2ctx(x_vertices),
                        y_vertices=np2ctx(y_vertices))

        # Try some particles inside
        parttest = xp.Particles(
                        _context=context,
                        p0c=6500e9,
                        x=x_vertices*0.99,
                        y=y_vertices*0.99)
        aper.track(parttest)
        assert np.allclose(ctx2np(parttest.state), 1)

        # Try some particles outside
        parttest = xp.Particles(
                        _context=context,
                        p0c=6500e9,
                        x=x_vertices*1.01,
                        y=y_vertices*1.01)
        aper.track(parttest)
        assert np.allclose(ctx2np(parttest.state), 0)

