import numpy as np
import xtrack as xt
import xobjects as xo
import xpart as xp

import ducktrack as dtk

from scipy.stats import linregress

from xobjects.context import available

def test_constructor():

    for ctx in xo.context.get_test_contexts():
        print(f"Test {ctx.__class__}")

        elements = [
            xt.Drift(_context=ctx),
            xt.Multipole(_context=ctx, knl=[2, 3]),
            xt.RFMultipole(_context=ctx, knl=[2]),
            xt.Cavity(voltage=3.),
            xt.SRotation(angle=4),
            xt.XYShift(dx=1),
            xt.DipoleEdge(h=1),
            xt.LimitRect(min_x=5),
            xt.LimitRectEllipse(max_x=6),
            xt.LimitEllipse(a=10),
            xt.LimitRacetrack(min_x=2),
            xt.LimitPolygon(x_vertices=[1,-1,-1,1], y_vertices=[1,1,-1,-1]),
            xt.Elens(inner_radius=0.1),
            xt.Wire(wire_current=3.)
        ]

        # test to_dict / from_dict
        for ee in elements:
            dd = ee.to_dict()
            nee = ee.__class__.from_dict(dd, _context=ctx)
            # Check that the two objects are bitwise identical
            if not isinstance(ctx, xo.ContextCpu):
                ee._move_to(_context=xo.ContextCpu())
                nee._move_to(_context=xo.ContextCpu())
            assert (ee._xobject._buffer.buffer[ee._xobject._offset:ee._xobject._size]
                    - nee._xobject._buffer.buffer[
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

        particles = xp.Particles.from_dict(dtk_particle.to_dict(),
                                           _context=ctx)

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

        particles = xp.Particles.from_dict(dtk_particle.to_dict(),
                                           _context=ctx)

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

        particles = xp.Particles.from_dict(dtk_particle.to_dict(),
                                           _context=ctx)

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

        particles = xp.Particles.from_dict(dtk_particle.to_dict(),
                                           _context=ctx)

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


def test_linear_transfer_uncorrelated_damping():
    for ctx in xo.context.get_test_contexts():
        print(f"Test {ctx.__class__}")

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
        damping_rate_x = 5E-4
        damping_rate_y = 1E-3
        damping_rate_s = 2E-3

        dtk_particle = dtk.TestParticles(
                p0c=25.92e9,
                x=1e-3,
                px=1e-5,
                y=-2e-3,
                py=-1.5e-5,
                zeta=2.,
                delta=2E-4)

        particles = xp.Particles.from_dict(dtk_particle.to_dict(),
                                           _context=ctx)


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
        y_ref_0 = y_ref_0, py_ref_0 = py_ref_0, y_ref_1 = y_ref_1, py_ref_1 = py_ref_1,
        damping_rate_x = damping_rate_x,damping_rate_y = damping_rate_y,damping_rate_s = damping_rate_s)
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
        y_ref_0 = y_ref_0, py_ref_0 = py_ref_0, y_ref_1 = y_ref_1, py_ref_1 = py_ref_1,
        damping_rate_x = damping_rate_x,damping_rate_y = damping_rate_y,damping_rate_s = damping_rate_s)
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

def test_linear_transfer_uncorrelated_damping_rate():
    for ctx in xo.context.get_test_contexts():
        print(f"Test {ctx.__class__}")

        alpha_x_0 = -0.5
        beta_x_0 = 100.0
        alpha_y_0 = -0.4
        beta_y_0 = 8.0
        Q_x = 0.18
        Q_y = 0.22
        beta_s = 856.9
        Q_s = 0.0015
        damping_rate_x = 5E-4
        damping_rate_y = 1E-3
        damping_rate_s = 2E-3
        energy = 45.6
        equ_emit_x = 0.3E-9
        equ_emit_y = 1E-12
        equ_length = 3.5E-3
        equ_delta = 3.8E-4
        beta_s = equ_length/equ_delta
        equ_emit_s = equ_length*equ_delta

        particles = xp.Particles(_context=ctx,
                    x=[10*np.sqrt(equ_emit_x*beta_x_0)],
                    y=[10*np.sqrt(equ_emit_y*beta_y_0)],
                    zeta=[10*np.sqrt(equ_emit_s*beta_s)],
                    p0c=energy*1E9)


        arc = xt.LinearTransferMatrix(_context=ctx,
        alpha_x_0=alpha_x_0, beta_x_0=beta_x_0,
        alpha_x_1=alpha_x_0, beta_x_1=beta_x_0,
        alpha_y_0=alpha_y_0, beta_y_0=beta_y_0,
        alpha_y_1=alpha_y_0, beta_y_1=beta_y_0,
        Q_x=Q_x, Q_y=Q_y,
        beta_s=beta_s, Q_s=Q_s,
        damping_rate_x = damping_rate_x,damping_rate_y = damping_rate_y,damping_rate_s = damping_rate_s)

        gamma_x = (1.0+alpha_x_0**2)/beta_x_0
        gamma_y = (1.0+alpha_y_0**2)/beta_y_0
        n_turns = int(1E4)
        emit_x = np.zeros(n_turns,dtype=float)
        emit_y = np.zeros_like(emit_x)
        emit_s = np.zeros_like(emit_x)
        ctx2np = ctx.nparray_from_context_array
        for turn in range(n_turns):
            arc.track(particles)
            emit_x[turn] = ctx2np(0.5*(gamma_x*particles.x[0]**2
                 + 2*alpha_x_0*particles.x[0]*particles.px[0]
                 + beta_x_0*particles.px[0]**2))
            emit_y[turn] = ctx2np(0.5*(gamma_y*particles.y[0]**2+2*alpha_y_0*particles.y[0]*particles.py[0]+beta_y_0*particles.py[0]**2))
            emit_s[turn] = ctx2np(0.5*(particles.zeta[0]**2/beta_s+beta_s*particles.delta[0]**2))
        turns = np.arange(n_turns)
        fit_x = linregress(turns,np.log(emit_x))
        fit_y = linregress(turns,np.log(emit_y))
        fit_s = linregress(turns,np.log(emit_s))

        assert np.isclose(damping_rate_x,-fit_x.slope, rtol=1e-3, atol=1e-10)
        assert np.isclose(damping_rate_y,-fit_y.slope, rtol=1e-3, atol=1e-10)
        assert np.isclose(damping_rate_s,-fit_s.slope, rtol=1e-3, atol=1e-10)

def test_linear_transfer_uncorrelated_damping_equilibrium():
    for ctx in xo.context.get_test_contexts():
        print(f"Test {ctx.__class__}")

        alpha_x_0 = -0.5
        beta_x_0 = 100.0
        alpha_y_0 = -0.4
        beta_y_0 = 8.0
        Q_x = 0.18
        Q_y = 0.22
        beta_s = 856.9
        Q_s = 0.015
        damping_rate_x = 5E-4
        damping_rate_y = 1E-3
        damping_rate_s = 2E-3
        energy = 45.6
        equ_emit_x = 0.3E-9
        equ_emit_y = 1E-12
        equ_length = 3.5E-3
        equ_delta = 3.8E-4
        beta_s = equ_length/equ_delta
        equ_emit_s = equ_length*equ_delta

        npart = int(1E3)
        particles = xp.Particles(_context=ctx,
                    x=np.random.randn(npart)*np.sqrt(equ_emit_x*beta_x_0),
                    px=np.random.randn(npart)*np.sqrt(equ_emit_x/beta_x_0),
                    y=np.random.randn(npart)*np.sqrt(equ_emit_y*beta_y_0),
                    py=np.random.randn(npart)*np.sqrt(equ_emit_y/beta_y_0),
                    zeta=np.random.randn(npart)*np.sqrt(equ_emit_s*beta_s),
                    delta=np.random.randn(npart)*np.sqrt(equ_emit_s/beta_s),
                    p0c=energy*1E9)
        particles._init_random_number_generator();


        arc = xt.LinearTransferMatrix(_context=ctx,
        alpha_x_0=alpha_x_0, beta_x_0=beta_x_0,
        alpha_x_1=alpha_x_0, beta_x_1=beta_x_0,
        alpha_y_0=alpha_y_0, beta_y_0=beta_y_0,
        alpha_y_1=alpha_y_0, beta_y_1=beta_y_0,
        Q_x=Q_x, Q_y=Q_y,
        beta_s=beta_s, Q_s=Q_s,
        damping_rate_x = damping_rate_x,damping_rate_y = damping_rate_y,damping_rate_s = damping_rate_s,
        equ_emit_x = equ_emit_x, equ_emit_y = equ_emit_y, equ_emit_s = equ_emit_s)

        gamma_x = (1.0+alpha_x_0**2)/beta_x_0
        gamma_y = (1.0+alpha_y_0**2)/beta_y_0
        n_turns = int(1E3)
        emit_x = np.zeros(n_turns,dtype=float)
        emit_y = np.zeros_like(emit_x)
        emit_s = np.zeros_like(emit_x)
        ctx2np = ctx.nparray_from_context_array
        for turn in range(n_turns):
            arc.track(particles)
            emit_x[turn] = 0.5*np.average(ctx2np(gamma_x*particles.x**2+2*alpha_x_0*particles.x*particles.px+beta_x_0*particles.px**2))
            emit_y[turn] = 0.5*np.average(ctx2np(gamma_y*particles.y**2+2*alpha_y_0*particles.y*particles.py+beta_y_0*particles.py**2))
            emit_s[turn] = 0.5*np.average(ctx2np(particles.zeta**2/beta_s+beta_s*particles.delta**2))
        turns = np.arange(n_turns)
        equ_emit_x_0 = np.average(emit_x)
        equ_emit_y_0 = np.average(emit_y)
        equ_emit_s_0 = np.average(emit_s)

        assert np.isclose(equ_emit_x,equ_emit_x_0, rtol=1e-1, atol=1e-10)
        assert np.isclose(equ_emit_y,equ_emit_y_0, rtol=1e-1, atol=1e-10)
        assert np.isclose(equ_emit_s,equ_emit_s_0, rtol=1e-1, atol=1e-10)


def test_cavity():

    for context in xo.context.get_test_contexts():
        print(f"Test {context.__class__}")

        cav = xt.Cavity(_context=context, frequency=0, lag=90, voltage=30)
        part = xp.Particles(p0c=1e9, delta=[0, 1e-2], zeta=[0, 0.2], _context=context)
        part0 = part.copy(_context=xo.ContextCpu())

        cav.track(part)

        part = part.copy(_context=xo.ContextCpu())

        assert np.allclose(part.energy,
                                part0.energy+cav.voltage, atol=5e-7, rtol=0)

        Pc = np.sqrt(part.energy**2 - part.mass0**2)
        delta = Pc/part.p0c - 1
        beta = Pc/part.energy

        tau0 = part0.zeta/(part0.beta0 * part0.rvv)
        tau = part.zeta/(part.beta0 * part.rvv)

        assert np.allclose(part.delta, delta, atol=1e-14, rtol=0)
        assert np.allclose(part.rpp, 1/(1+delta), atol=1e-14, rtol=0)
        assert np.allclose(part.rvv, beta/part.beta0, atol=1e-14, rtol=0)
        assert np.allclose(tau, tau0, atol=1e-14, rtol=0)
        assert np.allclose((part.ptau - part0.ptau) * part0.p0c, 30, atol=1e-9, rtol=0)



