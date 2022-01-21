import numpy as np
import xtrack as xt
import xobjects as xo
import xpart as xp

import ducktrack as dtk

from xobjects.context import available

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

def test_linear_transfer():
    for ctx in xo.context.get_test_contexts():
        print(f"Test {ctx.__class__}")

        dtk_particle = dtk.TestParticles(
                p0c=25.92e9,
                x=1e-3,
                px=1e-5,
                y=-2e-3,
                py=-1.5e-5,
                zeta=2.,
                delta=2E-4)

        particles = xp.Particles(_context=ctx,
                                 **dtk_particle.to_dict())

        alpha_x_0 = -0.5
        beta_x_0 = 100.0
        disp_x_0 = 1.8
        alpha_x_1 = 2.1
        beta_x_1 = 2.0
        disp_x_1 = 3.3
        alpha_y_0 = -0.4
        beta_y_0 = 8.0
        disp_y_0 = -0.2
        alpha_y_1 = 0.7
        beta_y_1 = 0.3
        disp_y_1 = -1.9
        Q_x = 0.27
        Q_y = 0.34
        beta_s = 856.9
        Q_s = 0.001
        energy_ref_increment = 1.2E9
        energy_increment = 4.8E8
        x_ref_0 = -5E-3
        px_ref_0 = 6E-4
        x_ref_1 = 2E-2
        px_ref_1 = -5E-5
        y_ref_0 = -9E-2
        py_ref_0 = 1E-4
        y_ref_1 = 4E-2
        py_ref_1 = 5E-4

        arc = xt.LinearTransferMatrix(_context=ctx,
        alpha_x_0=alpha_x_0, beta_x_0=beta_x_0, disp_x_0=disp_x_0,
        alpha_x_1=alpha_x_1, beta_x_1=beta_x_1, disp_x_1=disp_x_1,
        alpha_y_0=alpha_y_0, beta_y_0=beta_y_0, disp_y_0=disp_y_0,
        alpha_y_1=alpha_y_1, beta_y_1=beta_y_1, disp_y_1=disp_y_1,
        Q_x=Q_x, Q_y=Q_y,
        beta_s=beta_s, Q_s=Q_s,
        chroma_x=0.0, chroma_y=0.0,
        detx_x=0.0, detx_y=0.0, dety_y=0.0, dety_x=0.0,
        energy_ref_increment=energy_ref_increment,energy_increment=energy_increment,
        x_ref_0 = x_ref_0, px_ref_0 = px_ref_0, x_ref_1 = x_ref_1, px_ref_1 = px_ref_1,
        y_ref_0 = y_ref_0, py_ref_0 = py_ref_0, y_ref_1 = y_ref_1, py_ref_1 = py_ref_1)
        arc.track(particles)

        dtk_arc = dtk.elements.LinearTransferMatrix(alpha_x_0=alpha_x_0, beta_x_0=beta_x_0, disp_x_0=disp_x_0,
        alpha_x_1=alpha_x_1, beta_x_1=beta_x_1, disp_x_1=disp_x_1,
        alpha_y_0=alpha_y_0, beta_y_0=beta_y_0, disp_y_0=disp_y_0,
        alpha_y_1=alpha_y_1, beta_y_1=beta_y_1, disp_y_1=disp_y_1,
        Q_x=Q_x, Q_y=Q_y,
        beta_s=beta_s, Q_s=Q_s,
        chroma_x=0.0, chroma_y=0.0,
        detx_x=0.0, detx_y=0.0, dety_y=0.0, dety_x=0.0,
        energy_ref_increment=energy_ref_increment,energy_increment=energy_increment,
        x_ref_0 = x_ref_0, px_ref_0 = px_ref_0, x_ref_1 = x_ref_1, px_ref_1 = px_ref_1,
        y_ref_0 = y_ref_0, py_ref_0 = py_ref_0, y_ref_1 = y_ref_1, py_ref_1 = py_ref_1)
        dtk_arc.track(dtk_particle)

        assert np.isclose(ctx.nparray_from_context_array(particles.x)[0],
                          dtk_particle.x, rtol=1e-14, atol=1e-14)
        assert np.isclose(ctx.nparray_from_context_array(particles.px)[0],
                          dtk_particle.px, rtol=1e-14, atol=1e-14)
        assert np.isclose(ctx.nparray_from_context_array(particles.y)[0],
                          dtk_particle.y, rtol=1e-14, atol=1e-14)
        assert np.isclose(ctx.nparray_from_context_array(particles.py)[0],
                          dtk_particle.py, rtol=1e-14, atol=1e-14)
        assert np.isclose(ctx.nparray_from_context_array(particles.zeta)[0],
                          dtk_particle.zeta, rtol=1e-14, atol=1e-14)
        assert np.isclose(ctx.nparray_from_context_array(particles.delta)[0],
                          dtk_particle.delta, rtol=1e-14, atol=1e-14)


def test_linear_transfer_chroma_detuning():
    for ctx in xo.context.get_test_contexts():
        print(f"Test {ctx.__class__}")

        dtk_particle = dtk.TestParticles(
                p0c=25.92e9,
                x=1e-3,
                px=1e-5,
                y=-2e-3,
                py=-1.5e-5,
                zeta=2.,
                delta=2E-4)

        particles = xp.Particles(_context=ctx,
                                 **dtk_particle.to_dict())

        alpha_x_0 = -0.5
        beta_x_0 = 100.0
        disp_x_0 = 1.8
        alpha_x_1 = 2.1
        beta_x_1 = 2.0
        disp_x_1 = 3.3
        alpha_y_0 = -0.4
        beta_y_0 = 8.0
        disp_y_0 = -0.2
        alpha_y_1 = 0.7
        beta_y_1 = 0.3
        disp_y_1 = -1.9
        Q_x = 0.27
        Q_y = 0.34
        beta_s = 856.9
        Q_s = 0.001
        energy_ref_increment = 1.2E9
        energy_increment = 4.8E8
        x_ref_0 = -5E-3
        px_ref_0 = 6E-4
        x_ref_1 = 2E-2
        px_ref_1 = -5E-5
        y_ref_0 = -9E-2
        py_ref_0 = 1E-4
        y_ref_1 = 4E-2
        py_ref_1 = 5E-4
        chroma_x=8.0
        chroma_y=-5.0
        detx_x = 1E-3
        detx_y = -2E-4
        dety_y = -6E-4
        dety_x = 3E-3

        arc = xt.LinearTransferMatrix(_context=ctx,
        alpha_x_0=alpha_x_0, beta_x_0=beta_x_0, disp_x_0=disp_x_0,
        alpha_x_1=alpha_x_1, beta_x_1=beta_x_1, disp_x_1=disp_x_1,
        alpha_y_0=alpha_y_0, beta_y_0=beta_y_0, disp_y_0=disp_y_0,
        alpha_y_1=alpha_y_1, beta_y_1=beta_y_1, disp_y_1=disp_y_1,
        Q_x=Q_x, Q_y=Q_y,
        beta_s=beta_s, Q_s=Q_s,
        chroma_x=chroma_x, chroma_y=chroma_y,
        detx_x=detx_x, detx_y=detx_y, dety_y=dety_y, dety_x=dety_x,
        energy_ref_increment=energy_ref_increment,energy_increment=energy_increment,
        x_ref_0 = x_ref_0, px_ref_0 = px_ref_0, x_ref_1 = x_ref_1, px_ref_1 = px_ref_1,
        y_ref_0 = y_ref_0, py_ref_0 = py_ref_0, y_ref_1 = y_ref_1, py_ref_1 = py_ref_1)
        arc.track(particles)

        dtk_arc = dtk.elements.LinearTransferMatrix(alpha_x_0=alpha_x_0, beta_x_0=beta_x_0, disp_x_0=disp_x_0,
        alpha_x_1=alpha_x_1, beta_x_1=beta_x_1, disp_x_1=disp_x_1,
        alpha_y_0=alpha_y_0, beta_y_0=beta_y_0, disp_y_0=disp_y_0,
        alpha_y_1=alpha_y_1, beta_y_1=beta_y_1, disp_y_1=disp_y_1,
        Q_x=Q_x, Q_y=Q_y,
        beta_s=beta_s, Q_s=Q_s,
        chroma_x=chroma_x, chroma_y=chroma_y,
        detx_x=detx_x, detx_y=detx_y, dety_y=dety_y, dety_x=dety_x,
        energy_ref_increment=energy_ref_increment,energy_increment=energy_increment,
        x_ref_0 = x_ref_0, px_ref_0 = px_ref_0, x_ref_1 = x_ref_1, px_ref_1 = px_ref_1,
        y_ref_0 = y_ref_0, py_ref_0 = py_ref_0, y_ref_1 = y_ref_1, py_ref_1 = py_ref_1)
        dtk_arc.track(dtk_particle)

        assert np.isclose(ctx.nparray_from_context_array(particles.x)[0],
                          dtk_particle.x, rtol=1e-14, atol=1e-14)
        assert np.isclose(ctx.nparray_from_context_array(particles.px)[0],
                          dtk_particle.px, rtol=1e-14, atol=1e-14)
        assert np.isclose(ctx.nparray_from_context_array(particles.y)[0],
                          dtk_particle.y, rtol=1e-14, atol=1e-14)
        assert np.isclose(ctx.nparray_from_context_array(particles.py)[0],
                          dtk_particle.py, rtol=1e-14, atol=1e-14)
        assert np.isclose(ctx.nparray_from_context_array(particles.zeta)[0],
                          dtk_particle.zeta, rtol=1e-14, atol=1e-14)
        assert np.isclose(ctx.nparray_from_context_array(particles.delta)[0],
                          dtk_particle.delta, rtol=1e-14, atol=1e-14)





