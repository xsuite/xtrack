# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np
import pytest

import xtrack as xt
import xobjects as xo
import xpart as xp
from xobjects.test_helpers import for_all_test_contexts
from xtrack.beam_elements.elements import _angle_from_trig

import ducktrack as dtk

from scipy.stats import linregress

@for_all_test_contexts
def test_constructor(test_context):
    elements = [
        xt.Drift(_context=test_context),
        xt.Marker(_context=test_context),
        xt.Multipole(_context=test_context, knl=[2, 3]),
        xt.RFMultipole(_context=test_context, knl=[2]),
        xt.Cavity(_context=test_context, voltage=3.),
        xt.SRotation(_context=test_context, angle=4),
        xt.XRotation(_context=test_context, angle=1.8),
        xt.YRotation(_context=test_context, angle=2.4),
        xt.ZetaShift(_context=test_context, dzeta=3E-4),
        xt.XYShift(_context=test_context, dx=1),
        xt.DipoleEdge(_context=test_context, h=1),
        xt.LimitRect(_context=test_context, min_x=5),
        xt.LimitRectEllipse(_context=test_context, max_x=6),
        xt.LimitEllipse(_context=test_context, a=10),
        xt.LimitRacetrack(_context=test_context, min_x=-3, max_x=4,
                           min_y=2, max_y=3, a=0.2, b=0.3),
        xt.LimitPolygon(_context=test_context, x_vertices=[1,-1,-1,1], y_vertices=[1,1,-1,-1]),
        xt.Elens(_context=test_context, inner_radius=0.1),
        xt.Wire(_context=test_context, current=3.),
        xt.Exciter(_context=test_context, knl=[1], samples=[1,2,3],
                   sampling_frequency=1e3),
    ]

    # test to_dict / from_dict
    for ee in elements:
        dd = ee.to_dict()
        nee = ee.__class__.from_dict(dd, _context=test_context)
        # Check that the two objects are bitwise identical
        if not isinstance(test_context, xo.ContextCpu):
            ee.move(_context=xo.ContextCpu())
            nee.move(_context=xo.ContextCpu())
        assert (ee._xobject._buffer.buffer[ee._xobject._offset:ee._xobject._size]
                - nee._xobject._buffer.buffer[
                    nee._xobject._offset:nee._xobject._size]).sum() == 0


@pytest.mark.parametrize(
    'element_cls,args',
    [
        (xt.SRotation, {'angle': 8, 'sin_z': 0.4}),
        (xt.XRotation, {'angle': 8, 'sin_angle': 0.4}),
        (xt.YRotation, {'angle': 8, 'sin_angle': 0.4}),
        (xt.SRotation, {'angle': 8, 'cos_z': 0.4}),
        (xt.XRotation, {'angle': 8, 'cos_angle': 0.4}),
        (xt.YRotation, {'angle': 8, 'cos_angle': 0.4}),
        (xt.XRotation, {'angle': 8, 'tan_angle': 0.4}),
        (xt.YRotation, {'angle': 8, 'tan_angle': 0.4}),
    ],
)
def test_rotations_constructors_on_inconsistent_input(element_cls, args):
    with pytest.raises(ValueError):
        element_cls(**args)


@pytest.mark.parametrize(
    'cos,sin,tan,angle',
    [
        (np.cos(0.03), np.sin(0.03), np.tan(0.03), 0.03),
        (np.cos(0.01), np.sin(0.01), None, 0.01),
        (None, np.sin(0.02), np.tan(0.02), 0.02),
        (np.cos(0.03), None, np.tan(0.03), 0.03),
        (None, None, None, None),
        (0.1, None, None, None),
        (None, None, 0.1, None),
        (np.cos(0.02), np.sin(0.04), None, None),
        (None, np.sin(0.04), np.tan(0.02), None),
        (np.cos(0.02), None, np.tan(0.04), None),
    ]
)
def test__angle_from_trig(cos, sin, tan, angle):
    should_fail = angle is None

    if should_fail:
        with pytest.raises(ValueError):
            _angle_from_trig(cos, sin, tan)
    else:
        result, cos_res, sin_res, tan_res = _angle_from_trig(cos, sin, tan)

        if cos is not None:
            assert cos == cos_res
        if sin is not None:
            assert sin == sin_res
        if tan is not None:
            assert tan == tan_res

        assert np.isclose(angle, result, atol=1e-13)


@for_all_test_contexts
def test_backtrack(test_context):
    elements = [
        xt.Drift(_context=test_context),
        xt.Multipole(_context=test_context, knl=[2, 3]),
        xt.RFMultipole(_context=test_context, knl=[2]),
        xt.ReferenceEnergyIncrease(_context=test_context, Delta_p0c=42),
        xt.Cavity(_context=test_context, voltage=3.),
        xt.SRotation(_context=test_context, angle=4),
        xt.XRotation(_context=test_context, angle=0.3),
        xt.YRotation(_context=test_context, angle=0.7),
        xt.ZetaShift(_context=test_context, dzeta=3E-4),
        xt.XYShift(_context=test_context, dx=1),
        xt.DipoleEdge(_context=test_context, h=1),
        xt.LimitRect(_context=test_context, min_x=5),
        xt.LimitRectEllipse(_context=test_context, max_x=6),
        xt.LimitEllipse(_context=test_context, a=10),
        xt.LimitRacetrack(_context=test_context, min_x=-3, max_x=4,
                           min_y=2, max_y=3, a=0.2, b=0.3),
        xt.LimitPolygon(_context=test_context, x_vertices=[1,-1,-1,1], y_vertices=[1,1,-1,-1]),
        xt.Elens(_context=test_context, inner_radius=0.1),
        xt.Exciter(_context=test_context, knl=[1], samples=[1,2,3], sampling_frequency=1e3),
    ]

    dtk_particle = dtk.TestParticles(
            p0c=25.92e9,
            x=1e-3,
            px=1e-5,
            y=-2e-3,
            py=-1.5e-5,
            delta=1e-2,
            zeta=1.)

    for element in elements:
        element_backtrack = element.get_backtrack_element(_context=test_context)

        # track forth and back
        new_particles = xp.Particles.from_dict(dtk_particle.to_dict(), _context=test_context)
        element.track(new_particles)
        element_backtrack.track(new_particles)

        # assert that nothing changed
        for k in 'x,px,y,py,zeta,delta'.split(','):
            assert np.isclose(test_context.nparray_from_context_array(getattr(new_particles, k))[0],
                      getattr(dtk_particle, k), rtol=1e-14, atol=1e-14)


@for_all_test_contexts
def test_arr2ctx(test_context):
    d = xt.Drift(_context=test_context)

    a = [1., 2., 3.]
    assert type(d._arr2ctx(a)) is test_context.nplike_array_type

    a = test_context.zeros(shape=(20,), dtype=np.int64)
    assert type(d._arr2ctx(a)) is test_context.nplike_array_type
    assert (type(d._arr2ctx(a[1])) is int
            or (type(d._arr2ctx(a[1])) is test_context.nplike_array_type
                and d._arr2ctx(a[1]).shape == ()))

    a = np.array([1., 2., 3.])
    assert type(d._arr2ctx(a)) is test_context.nplike_array_type
    assert type(d._arr2ctx(a[1])) is float


@for_all_test_contexts
def test_drift(test_context):
    dtk_particle = dtk.TestParticles(
            p0c=25.92e9,
            x=1e-3,
            px=1e-5,
            y=-2e-3,
            py=-1.5e-5,
            delta=1e-2,
            zeta=1.)

    particles = xp.Particles.from_dict(dtk_particle.to_dict(),
                                       _context=test_context)

    drift = xt.Drift(_context=test_context, length=10.)
    drift.track(particles)

    dtk_drift = dtk.elements.Drift(length=10.)
    dtk_drift.track(dtk_particle)

    assert np.isclose(test_context.nparray_from_context_array(particles.x)[0],
                      dtk_particle.x, rtol=1e-14, atol=1e-14)
    assert np.isclose(test_context.nparray_from_context_array(particles.y)[0],
                      dtk_particle.y, rtol=1e-14, atol=1e-14)
    assert np.isclose(test_context.nparray_from_context_array(particles.zeta)[0],
                      dtk_particle.zeta,
                      rtol=1e-14, atol=1e-14)


@for_all_test_contexts
def test_drift_exact(test_context):
    dtk_particle = dtk.TestParticles(
            p0c=25.92e9,
            x=1e-3,
            px=1e-5,
            y=-2e-3,
            py=-1.5e-5,
            delta=1e-2,
            zeta=1.)

    particles = xp.Particles.from_dict(dtk_particle.to_dict(),
                                       _context=test_context)

    drift = xt.Drift(_context=test_context, length=10.)
    line = xt.Line(elements=[drift])
    line.build_tracker(compile=False, _context=test_context)
    line.config.XTRACK_USE_EXACT_DRIFTS = True
    line.track(particles)

    dtk_drift = dtk.elements.DriftExact(length=10.)
    dtk_drift.track(dtk_particle)

    assert np.isclose(test_context.nparray_from_context_array(particles.x)[0],
                      dtk_particle.x, rtol=1e-14, atol=1e-14)
    assert np.isclose(test_context.nparray_from_context_array(particles.y)[0],
                      dtk_particle.y, rtol=1e-14, atol=1e-14)
    assert np.isclose(test_context.nparray_from_context_array(particles.zeta)[0],
                      dtk_particle.zeta,
                      rtol=1e-14, atol=1e-14)


@for_all_test_contexts
def test_marker(test_context):
    dtk_particle = dtk.TestParticles(
        p0c=25.92e9,
        x=1e-3,
        px=1e-5,
        y=-2e-3,
        py=-1.5e-5,
        delta=1e-2,
        zeta=1.)

    particles = xp.Particles.from_dict(dtk_particle.to_dict(),
                                       _context=test_context)

    marker = xt.Marker(_context=test_context)
    marker.track(particles)

    assert np.isclose(test_context.nparray_from_context_array(particles.zeta)[0],
                      dtk_particle.zeta,
                      rtol=1e-14, atol=1e-14)
    assert np.isclose(test_context.nparray_from_context_array(particles.x)[0],
                      dtk_particle.x, rtol=1e-14, atol=1e-14)
    assert np.isclose(test_context.nparray_from_context_array(particles.y)[0],
                      dtk_particle.y, rtol=1e-14, atol=1e-14)


@for_all_test_contexts
def test_elens(test_context):
    dtk_particle = dtk.TestParticles(
            p0c  = np.array([7000e9]),
            x    = np.array([1e-3]),
            px   = np.array([0.0]),
            y    = np.array([2.2e-3]),
            py   = np.array([0.0]),
            zeta = np.array([0.]))

    particles = xp.Particles.from_dict(dtk_particle.to_dict(),
                                       _context=test_context)

    elens = xt.Elens(_context=test_context,
                     inner_radius=1.1e-3,
                     outer_radius=2.2e-3,
                     elens_length=3.,
                     voltage=15e3,
                     current=5)

    elens.track(particles)

    dtk_elens = dtk.elements.Elens(
                   inner_radius=1.1e-3,
                   outer_radius=2.2e-3,
                   elens_length=3.,
                   voltage=15e3,
                   current=5)

    dtk_elens.track(dtk_particle)

    assert np.isclose(test_context.nparray_from_context_array(particles.px)[0],
                      dtk_particle.px, rtol=1e-2, atol=1e-2)
    assert np.isclose(test_context.nparray_from_context_array(particles.py)[0],
                      dtk_particle.py, rtol=1e-9, atol=1e-9)


@for_all_test_contexts
def test_elens_measured_radial(test_context):
    def compute_coef(r_measured, j_measured, r_1_new, r_2_new,
                     r_1_old, r_2_old, p_order = 13):
        new_r = r_measured*(r_2_new-r_1_new)/(r_2_old-r_1_old)
        new_j = j_measured*(r_2_old-r_1_old)/(r_2_new-r_1_new)

        product = new_r*new_j

        delta_r = new_r[2]-new_r[1]

        numerator = [];
        s = len(new_r)
        for i in range(s):
            numerator.append((delta_r*max(np.cumsum(product[0:i+1]))))
        L = np.cumsum(product)
        denominator = max(L)*delta_r
        f_r = np.array(numerator/denominator)
        r_selected = new_r[new_j != 0]
        f_selected = f_r[new_j != 0]
        coef = np.polyfit(r_selected, f_selected, p_order)
        return coef


    particle_ref = xp.Particles(
                    p0c=np.array([7000e9]),
                    x=np.array([1e-3]),
                    px=np.array([0.0]),
                    y=np.array([2.2e-3]),
                    py=np.array([0.0]),
                    zeta=np.array([0.]))
    particle_test = particle_ref.copy(_context=test_context)

    # polynomial fit parameters for constant radial density
    r     = np.linspace(0.20338983,12,60)
    j     = np.append(np.append(np.linspace(0,4,20)*0,
           np.linspace(4,8,20)/np.linspace(4,8,20)), np.linspace(8,12,20)*0)
    C     = compute_coef(r, j, 1.4, 2.8, 4.0, 8.0)

    elens_radial_profile = xt.Elens(current=5, inner_radius=1.4e-3,
                outer_radius=2.8e-3, elens_length=3, voltage=10e3,
                coefficients_polynomial=C, _context=test_context)

    elens_constant = xt.Elens(current=5, inner_radius=1.4e-3,
                    outer_radius=2.8e-3, elens_length=3, voltage=10e3,
                    )

    elens_radial_profile.track(particle_test)
    elens_constant.track(particle_ref)

    particle_test.move(_context=xo.ContextCpu())

    assert np.isclose(particle_test.px[0], particle_ref.px[0],
                      rtol=1e-2, atol=1e-2)
    assert np.isclose(particle_test.py[0], particle_ref.py[0],
                      rtol=1e-2, atol=1e-2)


@for_all_test_contexts
def test_wire(test_context):
    dtk_particle = dtk.TestParticles(
            p0c =np.array([7000e9]),
            x   =np.array([1e-3]),
            px  =np.array([0.0]),
            y   =np.array([2.2e-3]),
            py  =np.array([0.0]),
            zeta=np.array([0.]))

    particles = xp.Particles(_context=test_context,
                             **dtk_particle.to_dict())


    wire = xt.Wire(_context    =  test_context,
                   L_phy  =  1.3,
                   L_int  =  1.3,
                   current=  250,
                   xma    = -8e-3,
                   yma    = -10e-3)

    wire.track(particles)

    dtk_wire = dtk.elements.Wire(
                   L_phy  =  1.3,
                   L_int  =  1.3,
                   current=  250,
                   xma    = -8e-3,
                   yma    = -10e-3)

    dtk_wire.track(dtk_particle)

    assert np.isclose(test_context.nparray_from_context_array(particles.px)[0],
                      dtk_particle.px, rtol=1e-9, atol=1e-9)
    assert np.isclose(test_context.nparray_from_context_array(particles.py)[0],
                      dtk_particle.py, rtol=1e-9, atol=1e-9)


@for_all_test_contexts
def test_linear_transfer(test_context):
    dtk_particle = dtk.TestParticles(
            p0c=25.92e9,
            x=1e-3,
            px=1e-5,
            y=-2e-3,
            py=-1.5e-5,
            zeta=2.,
            delta=2E-4)

    particles = xp.Particles.from_dict(dtk_particle.to_dict(),
                                       _context=test_context)

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

    arc = xt.LinearTransferMatrix(_context=test_context,
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

    assert np.isclose(test_context.nparray_from_context_array(particles.x)[0],
                      dtk_particle.x, rtol=1e-14, atol=1e-14)
    assert np.isclose(test_context.nparray_from_context_array(particles.px)[0],
                      dtk_particle.px, rtol=1e-14, atol=1e-14)
    assert np.isclose(test_context.nparray_from_context_array(particles.y)[0],
                      dtk_particle.y, rtol=1e-14, atol=1e-14)
    assert np.isclose(test_context.nparray_from_context_array(particles.py)[0],
                      dtk_particle.py, rtol=1e-14, atol=1e-14)
    assert np.isclose(test_context.nparray_from_context_array(particles.zeta)[0],
                      dtk_particle.zeta, rtol=1e-14, atol=1e-14)
    assert np.isclose(test_context.nparray_from_context_array(particles.delta)[0],
                      dtk_particle.delta, rtol=1e-14, atol=1e-14)


@for_all_test_contexts
def test_linear_transfer_chroma_detuning(test_context):
    dtk_particle = dtk.TestParticles(
            p0c=25.92e9,
            x=1e-3,
            px=1e-5,
            y=-2e-3,
            py=-1.5e-5,
            zeta=2.,
            delta=2E-4)

    particles = xp.Particles.from_dict(dtk_particle.to_dict(),
                                       _context=test_context)

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

    arc = xt.LinearTransferMatrix(_context=test_context,
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

    assert np.isclose(test_context.nparray_from_context_array(particles.x)[0],
                      dtk_particle.x, rtol=1e-14, atol=1e-14)
    assert np.isclose(test_context.nparray_from_context_array(particles.px)[0],
                      dtk_particle.px, rtol=1e-14, atol=1e-14)
    assert np.isclose(test_context.nparray_from_context_array(particles.y)[0],
                      dtk_particle.y, rtol=1e-14, atol=1e-14)
    assert np.isclose(test_context.nparray_from_context_array(particles.py)[0],
                      dtk_particle.py, rtol=1e-14, atol=1e-14)
    assert np.isclose(test_context.nparray_from_context_array(particles.zeta)[0],
                      dtk_particle.zeta, rtol=1e-14, atol=1e-14)
    assert np.isclose(test_context.nparray_from_context_array(particles.delta)[0],
                      dtk_particle.delta, rtol=1e-14, atol=1e-14)


@for_all_test_contexts
def test_linear_transfer_uncorrelated_damping(test_context):
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
                                       _context=test_context)


    arc = xt.LinearTransferMatrix(_context=test_context,
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

    assert np.isclose(test_context.nparray_from_context_array(particles.x)[0],
                      dtk_particle.x, rtol=1e-14, atol=1e-14)
    assert np.isclose(test_context.nparray_from_context_array(particles.px)[0],
                      dtk_particle.px, rtol=1e-14, atol=1e-14)
    assert np.isclose(test_context.nparray_from_context_array(particles.y)[0],
                      dtk_particle.y, rtol=1e-14, atol=1e-14)
    assert np.isclose(test_context.nparray_from_context_array(particles.py)[0],
                      dtk_particle.py, rtol=1e-14, atol=1e-14)
    assert np.isclose(test_context.nparray_from_context_array(particles.zeta)[0],
                      dtk_particle.zeta, rtol=1e-14, atol=1e-14)
    assert np.isclose(test_context.nparray_from_context_array(particles.delta)[0],
                      dtk_particle.delta, rtol=1e-14, atol=1e-14)


@for_all_test_contexts
def test_linear_transfer_uncorrelated_damping_rate(test_context):
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

    particles = xp.Particles(_context=test_context,
                x=[10*np.sqrt(equ_emit_x*beta_x_0)],
                y=[10*np.sqrt(equ_emit_y*beta_y_0)],
                zeta=[10*np.sqrt(equ_emit_s*beta_s)],
                p0c=energy*1E9)


    arc = xt.LinearTransferMatrix(_context=test_context,
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
    ctx2np = test_context.nparray_from_context_array
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


@for_all_test_contexts
def test_linear_transfer_uncorrelated_damping_equilibrium(test_context):
    alpha_x_0 = 0.0
    beta_x_0 = 100.0
    alpha_y_0 = 0.0
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
    particles = xp.Particles(_context=test_context,
                x=np.random.randn(npart)*np.sqrt(equ_emit_x*beta_x_0),
                px=np.random.randn(npart)*np.sqrt(equ_emit_x/beta_x_0),
                y=np.random.randn(npart)*np.sqrt(equ_emit_y*beta_y_0),
                py=np.random.randn(npart)*np.sqrt(equ_emit_y/beta_y_0),
                zeta=np.random.randn(npart)*np.sqrt(equ_emit_s*beta_s),
                delta=np.random.randn(npart)*np.sqrt(equ_emit_s/beta_s),
                p0c=energy*1E9)
    particles._init_random_number_generator();


    arc = xt.LinearTransferMatrix(_context=test_context,
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
    ctx2np = test_context.nparray_from_context_array
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


@for_all_test_contexts
def test_linear_transfer_first_order_taylor_map(test_context):
    dtk_particle = dtk.TestParticles(
            p0c=25.92e9,
            x=1e-3,
            px=1e-5,
            y=-2e-3,
            py=-1.5e-5,
            zeta=2.,
            delta=2E-4)

    particles = xp.Particles.from_dict(dtk_particle.to_dict(),
                                       _context=test_context)

    m0 = np.arange(6,dtype=float)
    m1 = np.ones((6,6),dtype=float)
    for i in range(6):
        for j in range(6):
            m1[i,j] = 10*i+j
    arc = xt.FirstOrderTaylorMap(_context=test_context,m0 = m0, m1 = m1)
    arc.track(particles)

    dtk_arc = dtk.elements.FirstOrderTaylorMap(m0 = m0, m1 = m1)
    dtk_arc.track(dtk_particle)

    assert np.isclose(test_context.nparray_from_context_array(particles.x)[0],
                      dtk_particle.x, rtol=1e-14, atol=1e-14)
    assert np.isclose(test_context.nparray_from_context_array(particles.px)[0],
                      dtk_particle.px, rtol=1e-14, atol=1e-14)
    assert np.isclose(test_context.nparray_from_context_array(particles.y)[0],
                      dtk_particle.y, rtol=1e-14, atol=1e-14)
    assert np.isclose(test_context.nparray_from_context_array(particles.py)[0],
                      dtk_particle.py, rtol=1e-14, atol=1e-14)
    assert np.isclose(test_context.nparray_from_context_array(particles.zeta)[0],
                      dtk_particle.zeta, rtol=1e-14, atol=1e-14)
    assert np.isclose(test_context.nparray_from_context_array(particles.delta)[0],
                      dtk_particle.delta, rtol=1e-14, atol=1e-14)


@for_all_test_contexts
def test_cavity(test_context):
    cav = xt.Cavity(_context=test_context, frequency=0, lag=90, voltage=30)
    part = xp.Particles(p0c=1e9, delta=[0, 1e-2], zeta=[0, 0.2], _context=test_context)
    part0 = part.copy(_context=xo.ContextCpu())

    cav.track(part)

    part = part.copy(_context=xo.ContextCpu())

    assert np.allclose(part.energy,
                            part0.energy+cav.voltage, atol=5e-7, rtol=0)

    Pc = np.sqrt(part.energy**2 - part.mass0**2)
    delta = Pc/part.p0c - 1
    beta = Pc/part.energy

    tau0 = part0.zeta/(part0.beta0)
    tau = part.zeta/(part.beta0)

    assert np.allclose(part.delta, delta, atol=1e-14, rtol=0)
    assert np.allclose(part.rpp, 1/(1+delta), atol=1e-14, rtol=0)
    assert np.allclose(part.rvv, beta/part.beta0, atol=1e-14, rtol=0)
    assert np.allclose(tau, tau0, atol=1e-14, rtol=0)
    assert np.allclose((part.ptau - part0.ptau) * part0.p0c, 30, atol=1e-9, rtol=0)

@for_all_test_contexts
def test_exciter(test_context):
    fs = 2.99792458e8 # sampling frequency in Hz
    frev = 2.99792458e8 # revolution frequency in Hz
    k0l = -0.1 # this is scaled by the waveform
    signal = [1,2,3] # this is the waveform
    duration = 4/fs
    exciter = xt.Exciter(samples=signal, sampling_frequency=fs,
                        frev=frev, start_turn=0, knl=[k0l], duration=duration)

    line = xt.Line([exciter])
    tracker = line.build_tracker(_context=test_context)

    particles = xp.Particles(p0c=6.5e12, zeta=[0,-1,-2], _context=test_context)
    num_particles = len(particles.zeta)

    tracker.track(particles, num_turns=1)
    expected_px = np.array([0.1, 0.2, 0.3])
    particles.move(_context=xo.context_default)

    assert np.allclose(particles.px, expected_px)

    particles.move(_context=test_context)
    tracker.track(particles, num_turns=1)
    expected_px += np.array([0.2, 0.3, 0.1])
    particles.move(_context=xo.context_default)
    assert np.allclose(particles.px, expected_px)

    particles.move(_context=test_context)
    tracker.track(particles, num_turns=1)
    expected_px += np.array([0.3, 0.1, 0])
    particles.move(_context=xo.context_default)
    assert np.allclose(particles.px, expected_px)


test_source = r"""
/*gpufun*/
void test_function(TestElementData el,
                LocalParticle* part0,
                /*gpuglmem*/ double* b){

    double const a = TestElementData_get_a(el);

    //start_per_particle_block (part0->part)

        const int64_t ipart = part->ipart;
        double const val = b[ipart];

        LocalParticle_add_to_x(part, val + a);

    //end_per_particle_block
}

/*gpufun*/
void TestElement_track_local_particle(TestElementData el,
                LocalParticle* part0){

    double const a = TestElementData_get_a(el);

    //start_per_particle_block (part0->part)

        LocalParticle_set_x(part, a);

    //end_per_particle_block
}

"""


@for_all_test_contexts
def test_per_particle_kernel(test_context):
    class TestElement(xt.BeamElement):
        _xofields={
            'a': xo.Float64
        }

        _extra_c_sources = [test_source]

        _per_particle_kernels = {
            'test_kernel': xo.Kernel(
                c_name='test_function',
                args=[
                    xo.Arg(xo.Float64, pointer=True, name='b')
                ]),
        }

    el = TestElement(_context=test_context, a=10)

    # p = xp.Particles(p0c=1e9, x=[1,2,3], _context=test_context)
    # el.track(p)
    # p.move(_context=xo.ContextCpu())
    # assert np.all(p.x == [10,10,10])

    p = xp.Particles(p0c=1e9, x=[1, 2, 3], _context=test_context)
    b = p.x*0.5
    el.test_kernel(p, b=b)
    p.move(_context=xo.ContextCpu())
    assert np.all(p.x == np.array([11.5, 13, 14.5]))
