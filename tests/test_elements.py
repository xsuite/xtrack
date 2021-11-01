import numpy as np
import xtrack as xt
import xobjects as xo
import xline as xl
import xpart as xp

from xobjects.context import available

def test_drift():

    for ctx in xo.context.get_test_contexts():
        print(f"Test {ctx.__class__}")

        pyst_particle = xl.XlineTestParticles(
                p0c=25.92e9,
                x=1e-3,
                px=1e-5,
                y=-2e-3,
                py=-1.5e-5,
                zeta=2.)

        particles = xp.Particles(_context=ctx,
                                 **pyst_particle.to_dict())

        drift = xt.Drift(_context=ctx, length=10.)
        drift.track(particles)

        pyst_drift = xl.elements.Drift(length=10.)
        pyst_drift.track(pyst_particle)

        assert np.isclose(ctx.nparray_from_context_array(particles.x)[0],
                          pyst_particle.x, rtol=1e-14, atol=1e-14)
        assert np.isclose(ctx.nparray_from_context_array(particles.y)[0],
                          pyst_particle.y, rtol=1e-14, atol=1e-14)



def test_elens():

    for ctx in xo.context.get_test_contexts():
        print(f"Test {ctx.__class__}")

        pyst_particle = xl.XlineTestParticles(
                p0c=np.array([7000e9]),
                x=np.array([1e-3]),
                px=np.array([0.0]),
                y=np.array([2.2e-3]),
                py=np.array([0.0]),
                zeta=np.array([0.]))

        particles = xp.Particles(_context=ctx,
                                 **pyst_particle.to_dict())


        elens = xt.Elens(_context=ctx,
                       inner_radius=1.1e-3,
                       outer_radius=2.2e-3,
                       elens_length=3.,
                       voltage=15e3,
                       current=5)

        elens.track(particles)

        pyst_elens = xl.elements.Elens(inner_radius=1.1e-3,
                       outer_radius=2.2e-3,
                       elens_length=3.,
                       voltage=15e3,
                       current=5)

        pyst_elens.track(pyst_particle)

        assert np.isclose(ctx.nparray_from_context_array(particles.px)[0],
                          pyst_particle.px, rtol=1e-9, atol=1e-9)
        assert np.isclose(ctx.nparray_from_context_array(particles.py)[0],
                          pyst_particle.py, rtol=1e-9, atol=1e-9)
