import numpy as np

import xobjects as xo
import xtrack as xt

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
        parttest = xt.Particles(
                        _context=context,
                        p0c=6500e9,
                        x=x_vertices*0.99,
                        y=y_vertices*0.99)
        aper.track(parttest)
        assert np.allclose(ctx2np(parttest.state), 1)

        # Try some particles outside
        parttest = xt.Particles(
                        _context=context,
                        p0c=6500e9,
                        x=x_vertices*1.01,
                        y=y_vertices*1.01)
        aper.track(parttest)
        assert np.allclose(ctx2np(parttest.state), 0)

