from math import factorial

import numpy as np
import pytest
from scipy.constants import c

import xobjects as xo
import xtrack as xt
from xobjects.test_helpers import for_all_test_contexts
from xtrack.linear_normal_form import S


@for_all_test_contexts
@pytest.mark.parametrize('components', ['normal', 'skew', 'both'])
@pytest.mark.parametrize('q0,charge_ratio,mass_ratio', [
    (1, 1., 1.), (2, 1., 1.), (-2, 1., 1.), (82, 0.5, 1.25)])
def test_rfmultipole_kicks_and_symplecticity(
        test_context, components, q0, charge_ratio, mass_ratio):
    element = xt.RFMultipole(knl=np.zeros(6), ksl=np.zeros(6),
                            frequency=400e6, voltage=2e6, phase=0.8)
    line = xt.Line(elements=[element])
    line.build_tracker(_context=test_context)
    p0 = xt.Particles(_context=test_context, p0c=1e9, q0=q0,
                      charge_ratio=charge_ratio, mass_ratio=mass_ratio,
                      x=0.13, y=-0.09, px=1e-4, py=-2e-4,
                      zeta=0.05, delta=1e-3)
    beta0 = float(p0.beta0[0])
    chi = charge_ratio / mass_ratio
    k = 2 * np.pi * element.frequency / c
    w = 0.13 - 0.09j
    phase_n = 0.6 - k * 0.05 / beta0
    phase_s = -0.4 - k * 0.05 / beta0
    cavity_kick = (abs(q0) * chi * element.voltage / (beta0 * 1e9)
                   * np.sin(element.phase - k * 0.05 / beta0))

    for order in range(6):
        bn = 0.1 * factorial(order) if components != 'skew' else 0.
        an = -0.07 * factorial(order) if components != 'normal' else 0.
        element.knl[:] = 0
        element.ksl[:] = 0
        element.knl[order] = bn
        element.ksl[order] = an
        element.phase_n[order] = 0.6
        element.phase_s[order] = -0.4

        # All three kicks are derivatives of the same multipole potential,
        # proportional to (x + iy)**(order + 1) / (order + 1)!.
        transverse = w**order / factorial(order)
        potential = w**(order + 1) / factorial(order + 1)
        dpx = -chi * (bn * np.cos(phase_n) * transverse.real
                      - an * np.cos(phase_s) * transverse.imag)
        dpy = chi * (bn * np.cos(phase_n) * transverse.imag
                     + an * np.cos(phase_s) * transverse.real)
        dpzeta = -chi * k / beta0 * (
            bn * np.sin(phase_n) * potential.real
            - an * np.sin(phase_s) * potential.imag)

        p = p0.copy()
        line.track(p)
        xo.assert_allclose(p.px, p0.px + dpx, rtol=1e-12, atol=1e-15)
        xo.assert_allclose(p.py, p0.py + dpy, rtol=1e-12, atol=1e-15)
        xo.assert_allclose(p.pzeta, p0.pzeta + dpzeta + cavity_kick,
                           rtol=1e-12, atol=1e-15)

        matrix = line.get_R_matrix(particle_on_co=p0.copy())['R_matrix']
        xo.assert_allclose(matrix.T @ S @ matrix, S, rtol=0, atol=2e-8)

        line.track(p, backtrack=True)
        for field in ['x', 'px', 'y', 'py', 'zeta', 'pzeta']:
            xo.assert_allclose(getattr(p, field), getattr(p0, field),
                               rtol=0, atol=1e-14)
