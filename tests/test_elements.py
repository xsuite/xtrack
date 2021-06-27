import numpy as np
import xtrack as xt
import xobjects as xo
import xline as xl

from xobjects.context import available

def test_drift():

    for CTX in xo.ContextCpu, xo.ContextPyopencl, xo.ContextCupy:
        if CTX not in available:
            continue

        print(f"Test {CTX}")
        ctx = CTX()

        pyst_particle = xl.Particles(
                p0c=25.92e9,
                x=1e-3,
                px=1e-5,
                y=-2e-3,
                py=-1.5e-5,
                zeta=2.)

        part_dict = xt.pyparticles_to_xtrack_dict(pyst_particle)
        particles = xt.Particles(_context=ctx,
                                 **part_dict)

        drift = xt.Drift(_context=ctx, length=10.)
        drift.track(particles)

        pyst_drift = xl.elements.Drift(length=10.)
        pyst_drift.track(pyst_particle)

        assert np.isclose(ctx.nparray_from_context_array(particles.x)[0],
                          pyst_particle.x, rtol=1e-14, atol=1e-14)
        assert np.isclose(ctx.nparray_from_context_array(particles.y)[0],
                          pyst_particle.y, rtol=1e-14, atol=1e-14)
