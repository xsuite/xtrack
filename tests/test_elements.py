import numpy as np
import xtrack as xt
import xobjects as xo
import xpart as xp

import ducktrack as dtk

from xobjects.context import available

def test_constructor():

    for ctx in xo.context.get_test_contexts():
        print(f"Test {ctx.__class__}")

        elements = [
            xt.Drift(_context=ctx),
            xt.Multipole(_context=ctx, knl=[2, 3]),
            xt.RFMultipole(_context=ctx, knl=[2]),
            xt.Cavity(),
            xt.SRotation(),
            xt.XYShift(),
            xt.DipoleEdge(),
        ]

        # test to_dict / from_dict
        for ee in elements:
            dd = ee.to_dict()
            nee = ee.__class__.from_dict(dd, _context=ctx)
            # Check that the two objects are bitwise identical
            assert (ee.copy(_context=xo.ContextCpu())._xobject._buffer.buffer[
                      ee._xobject._offset:ee._xobject._size]
                    - nee.copy(_context=xo.ContextCpu())._xobject._buffer.buffer[
                        nee._xobject._offset:nee._xobject._size]).sum() == 0

def test_drift():

    for ctx in xo.context.get_test_contexts():
        print(f"Test {ctx.__class__}")

        dtk_particle = dtk.TestParticles(
                p0c=25.92e9,
                x=1e-3,
                px=1e-5,
                y=-2e-3,
                py=-1.5e-5,
                zeta=2.)

        particles = xp.Particles(_context=ctx,
                                 **dtk_particle.to_dict())

        drift = xt.Drift(_context=ctx, length=10.)
        drift.track(particles)

        dtk_drift = dtk.elements.Drift(length=10.)
        dtk_drift.track(dtk_particle)

        assert np.isclose(ctx.nparray_from_context_array(particles.x)[0],
                          dtk_particle.x, rtol=1e-14, atol=1e-14)
        assert np.isclose(ctx.nparray_from_context_array(particles.y)[0],
                          dtk_particle.y, rtol=1e-14, atol=1e-14)



def test_elens():

    for ctx in xo.context.get_test_contexts():
        print(f"Test {ctx.__class__}")

        dtk_particle = dtk.TestParticles(
                p0c=np.array([7000e9]),
                x=np.array([1e-3]),
                px=np.array([0.0]),
                y=np.array([2.2e-3]),
                py=np.array([0.0]),
                zeta=np.array([0.]))

        particles = xp.Particles(_context=ctx,
                                 **dtk_particle.to_dict())


        elens = xt.Elens(_context=ctx,
                       inner_radius=1.1e-3,
                       outer_radius=2.2e-3,
                       elens_length=3.,
                       voltage=15e3,
                       current=5)

        elens.track(particles)

        dtk_elens = dtk.elements.Elens(inner_radius=1.1e-3,
                       outer_radius=2.2e-3,
                       elens_length=3.,
                       voltage=15e3,
                       current=5)

        dtk_elens.track(dtk_particle)

        assert np.isclose(ctx.nparray_from_context_array(particles.px)[0],
                          dtk_particle.px, rtol=1e-9, atol=1e-9)
        assert np.isclose(ctx.nparray_from_context_array(particles.py)[0],
                          dtk_particle.py, rtol=1e-9, atol=1e-9)

        
        
def test_wire():

    for ctx in xo.context.get_test_contexts():
        print(f"Test {ctx.__class__}")

        dtk_particle = dtk.TestParticles(
                p0c =np.array([7000e9]),
                x   =np.array([1e-3]),
                px  =np.array([0.0]),
                y   =np.array([2.2e-3]),
                py  =np.array([0.0]),
                zeta=np.array([0.]))

        particles = xp.Particles(_context=ctx,
                                 **dtk_particle.to_dict())


        wire = xt.Wire(_context    =  ctx,
                       wire_L_phy  =  1.3,
                       wire_L_int  =  1.3,
                       wire_current=  250,
                       wire_xma    = -8e-3,
                       wire_yma    = -10e-3)

        wire.track(particles)

        dtk_wire = dtk.elements.Wire(
                       wire_L_phy  =  1.3,
                       wire_L_int  =  1.3,
                       wire_current=  250,
                       wire_xma    = -8e-3,
                       wire_yma    = -10e-3)

        dtk_wire.track(dtk_particle)

        assert np.isclose(ctx.nparray_from_context_array(particles.px)[0],
                          dtk_particle.px, rtol=1e-9, atol=1e-9)
        assert np.isclose(ctx.nparray_from_context_array(particles.py)[0],
                          dtk_particle.py, rtol=1e-9, atol=1e-9)
        
        
def test_linked_arrays_in_multipole_and_rfmultipole():

    for ctx in xo.context.get_test_contexts():
        print(f"Test {ctx.__class__}")

        mult = xt.Multipole(_context=ctx, knl=[1,2,3,4], ksl=[10, 20, 30, 40])
        rfmult = xt.RFMultipole(_context=ctx, knl=[1,2,3,4], ksl=[10, 20, 30, 40],
                                frequency=10.)
        for m in [mult, rfmult]:
            assert np.allclose(ctx.nparray_from_context_array(m.bal),
                            [ 1., 10.,  2., 20.,  1.5 , 15.,
                                0.66666667,  6.66666667], rtol=0, atol=1e-8)

            m.knl[2:] = m.knl[2:] + 2
            assert np.allclose(ctx.nparray_from_context_array(m.bal),
                            [ 1., 10.,  2., 20.,  2.5 , 15.,
                                1.,  6.66666667], rtol=0, atol=1e-8)

            m.ksl[2:] = m.ksl[2:] + 20
            assert np.allclose(ctx.nparray_from_context_array(m.bal),
                            [ 1., 10.,  2., 20.,  2.5 , 25.,
                                1.,  10.], rtol=0, atol=1e-8)

            assert np.allclose(ctx.nparray_from_context_array(m.knl),
                            [1, 2, 5, 6], rtol=0, atol=1e-12)

            assert np.allclose(ctx.nparray_from_context_array(m.ksl),
                               [10, 20, 50, 60], rtol=0, atol=1e-12)

