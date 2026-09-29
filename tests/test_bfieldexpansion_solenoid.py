from math import factorial

import numpy as np
from numpy.polynomial import Polynomial
import pytest
import xobjects as xo
import xtrack as xt


@pytest.mark.parametrize('h', [0., 0.3])
@pytest.mark.parametrize('degree', [0, 1, 3, 7])
def test_ksol_retains_highest_coefficient(h, degree):
    coefficients = np.arange(1, degree + 2) / 10
    coefficients[0] = 0.5
    element = xt.BFieldExpansion(length=0.4, h=h, ksol=coefficients,
                                s_start=0.13, kscale=0.7)
    # User-visible shapes and degree describe the fields, not their integral.
    assert element.ksol.shape == (degree + 1,)
    assert element.knc.shape == element.ksc.shape == (1, degree + 1)
    assert element.deg == degree
    s_local = np.array([0., 0.2, 0.4])
    s = element.s_start + s_local
    x = np.array([0., -0.02, 0.03])
    y = np.array([0.01, -0.02, 0.03])

    # Exercise updates of the highest coefficient through zero and back.
    for last in (coefficients[-1], 0., -coefficients[-1]):
        element.ksol[-1] = last
        profile = Polynomial(np.asarray(element.ksol))
        for candidate in (element, element.copy(),
                          xt.BFieldExpansion.from_dict(element.to_dict())):
            field = candidate.get_field(x, 0., s_local)
            xo.assert_allclose(field['Bs'], 0.7 * profile(s) / (1 + h*x),
                               rtol=0, atol=2e-14)
            xo.assert_allclose(field['phi'], -0.7 * profile.integ()(s),
                               rtol=0, atol=2e-14)
            integral = profile.integ()
            xo.assert_allclose(candidate.ksoll,
                               0.7 * (integral(0.53) - integral(0.13)),
                               rtol=0, atol=2e-14)

        # Previously valid padded inputs must still give the same off-axis
        # fields, potentials and derivatives at a matched transverse order.
        padded = xt.BFieldExpansion(length=0.4, h=h, s_start=0.13, kscale=0.7,
            ksol=np.r_[np.asarray(element.ksol), 0.], num_phi=element.num_phi)
        actual, expected = element.get_field(x, y, s_local), padded.get_field(x, y, s_local)
        for name in actual.dtype.names:
            xo.assert_allclose(actual[name], expected[name], rtol=0, atol=2e-14)


@pytest.mark.parametrize('degree', [1, 3, 7])
def test_ksol_straight_analytic_field(degree):
    profile = Polynomial(np.arange(1, degree + 2) / 10)
    element = xt.BFieldExpansion(length=1., ksol=profile.coef)
    y, s = np.array([0.2, -0.3]), np.array([0.4, 0.7])
    # With zero transverse seeds this is a planar Maxwell expansion.
    bs = sum((-1)**k * y**(2*k) * profile.deriv(2*k)(s) / factorial(2*k)
             for k in range(degree//2 + 1))
    by = sum((-1)**(k+1) * y**(2*k+1) * profile.deriv(2*k+1)(s) / factorial(2*k+1)
             for k in range((degree+1)//2))
    ax = sum((-1)**(k+1) * y**(2*k+1) * profile.deriv(2*k)(s) / factorial(2*k+1)
             for k in range(degree//2 + 1))
    field = element.get_field(x=0.1, y=y, s_local=s)
    for name, expected in [('Bx', 0.), ('Bs', bs), ('By', by), ('Ax', ax)]:
        xo.assert_allclose(field[name], expected, rtol=0, atol=2e-14)


@pytest.mark.parametrize('h', [0., 0.3])
@pytest.mark.parametrize('degree', [0, 3, 7])
def test_ksol_highest_coefficient_environment(h, degree):
    env = xt.Environment()
    env['strength'] = 0.
    env.new('sol', xt.BFieldExpansion, length=0.5, h=h,
            ksol=[0.] * degree + ['strength'])
    element = env.get('sol')
    original_order = element.num_phi
    for strength in (0.5, -0.3):
        env['strength'] = strength
        field = element.get_field(x=0., y=0., s_local=0.4)
        assert field['Bs'] == pytest.approx(strength * 0.4**degree, abs=1e-14)
    env.set('sol', ksol=[0.] * degree + ['2*strength'])
    env['strength'] = 0.7
    assert element.get_field(0., 0., 0.4)['Bs'] == pytest.approx(1.4 * 0.4**degree)
    assert element.num_phi == original_order
    assert element.ksol.shape == (degree + 1,)


@pytest.mark.parametrize('pkin_const', [False, True])
def test_constant_ksol_against_exact_solenoid(pkin_const):
    ks, length = 0.5, 0.8
    initial = xt.Particles(p0c=1e9, x=[0.01, -0.02], y=[0.02, -0.01],
                           px=[0.03, -0.02], py=[-0.01, 0.04], delta=[-0.1, 0.2])
    reference = initial.copy()
    # Compare the uniform body with matching kinetic entrance momenta.
    # UniformSolenoid uses the symmetric gauge; BFieldExpansion has Ay=0.
    reference.ax = -0.5 * ks * reference.y
    reference.ay = 0.5 * ks * reference.x
    reference.px += reference.ax
    reference.py += reference.ay
    exact = xt.UniformSolenoid(length=length, ks=ks,
                              edge_entry_active=False, edge_exit_active=False)
    exact.track(reference)

    expansion = xt.BFieldExpansion(length=length, ksol=[ks], pkin_const=pkin_const)
    errors = []
    for steps in (8, 16, 32):
        expansion.num_integration_steps = steps
        particles = initial.copy()
        if not pkin_const:
            particles.ax = -ks * particles.y
            particles.px += particles.ax
        expansion.track(particles)
        errors.append(max(np.max(np.abs(getattr(particles, name) - getattr(reference, name)))
                          for name in ('x', 'kin_px', 'y', 'kin_py', 'zeta', 'delta')))
    assert errors[-1] < 1e-10
    xo.assert_allclose(np.array(errors[:-1]) / errors[1:], 16., rtol=0.1, atol=0)
