# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

import numpy as np
import pytest
import xobjects as xo
import xtrack as xt
from scipy.constants import c as clight
from xobjects.test_helpers import for_all_test_contexts

xo.settings.allow_kernel_compilation = True


@for_all_test_contexts
@pytest.mark.parametrize('h,k0', [
    (0.1, 0.1),
    (0.1, 0),
    (0, 0.1)
])
def test_bfieldexpansion_spin_constant_h_k0(test_context, h, k0):
    length = 1.0
    angle = h * length

    p0 = xt.Particles(
        mass0=xt.PROTON_MASS_EV,
        q0=1,
        p0c=1e9,
        x=1e-3,
        px=2e-4,
        y=-2e-3,
        py=-1e-4,
        delta=1e-3,
        zeta=1e-2,
        spin_x=0.2,
        spin_y=0.1,
        spin_z=0.3,
    )

    # Normalize spin
    spin_norm = np.sqrt(p0.spin_x[0]**2 + p0.spin_y[0]**2 + p0.spin_z[0]**2)
    p0.spin_x /= spin_norm
    p0.spin_y /= spin_norm
    p0.spin_z /= spin_norm

    p_bend = p0.copy()
    p_bfe = p0.copy()

    bend = xt.Bend(length=length, angle=angle, k0=k0)
    line_bend = xt.Line(elements=[bend])
    line_bend.particle_ref = p0.copy()
    line_bend.configure_spin('auto')
    line_bend.build_tracker(_context=test_context)
    line_bend.track(p_bend)

    bfe = xt.BFieldExpansion(
        length=length,
        h=h,
        k0=k0,
        num_integration_steps=2000,
    )
    line_bfe = xt.Line(elements=[bfe])
    line_bfe.particle_ref = p0.copy()
    line_bfe.configure_spin('auto')
    line_bfe.build_tracker(_context=test_context)
    line_bfe.track(p_bfe)

    p_bend.move(_context=xo.context_default)
    p_bfe.move(_context=xo.context_default)

    assert p_bend.state[0] == 1
    assert p_bfe.state[0] == 1

    for coord in ('x', 'px', 'y', 'py', 'zeta', 'delta',
                  'spin_x', 'spin_y', 'spin_z'):
        xo.assert_allclose(getattr(p_bfe, coord), getattr(p_bend, coord),
                           rtol=0, atol=5e-8)

    spin_norm_out = np.sqrt(
        p_bfe.spin_x[0]**2 + p_bfe.spin_y[0]**2 + p_bfe.spin_z[0]**2)
    xo.assert_allclose(spin_norm_out, 1.0, rtol=0, atol=1e-10)


@for_all_test_contexts
@pytest.mark.parametrize('case_name,particle_init,knc,ksc', [
    (
        'normal_s_dependent',
        dict(x=1e-3, px=2e-4, y=-2e-3, py=-1e-4, delta=1e-3, zeta=1e-2,
             spin_x=0.2, spin_y=0.1, spin_z=0.3),
        [[0.2, 0.01, -0.005],    # k0(s)
         [0.10, -0.04, 0.02]],   # k1(s)
        None,
    ),
    (
        'skew_s_dependent',
        dict(x=2e-3, px=-1e-4, y=1.5e-3, py=2e-4, delta=-5e-4, zeta=-1e-2,
             spin_x=0.3, spin_y=-0.2, spin_z=0.4),
        None,
        [[0.015, -0.008, 0.004],  # k0s(s)
         [0.08, 0.03, -0.015]],   # k1s(s)
    ),
])
def test_bfieldexpansion_spin_vs_splineboris(
        test_context, case_name, particle_init, knc, ksc):
    length = 1.0

    p0 = xt.Particles(
        mass0=xt.PROTON_MASS_EV,
        q0=1,
        p0c=1e9,
        **particle_init,
    )

    spin_norm = np.sqrt(p0.spin_x[0]**2 + p0.spin_y[0]**2 + p0.spin_z[0]**2)
    p0.spin_x /= spin_norm
    p0.spin_y /= spin_norm
    p0.spin_z /= spin_norm

    brho = p0.p0c[0] / (clight * p0.q0)

    def hermite_from_poly(coeffs, L):
        # coeffs for a0 + a1*s + a2*s^2 + a3*s^3
        a0 = coeffs[0]
        a1 = coeffs[1] if len(coeffs) > 1 else 0.0
        a2 = coeffs[2] if len(coeffs) > 2 else 0.0
        a3 = coeffs[3] if len(coeffs) > 3 else 0.0
        val_start = a0
        der_start = a1
        val_end = a0 + a1*L + a2*L**2 + a3*L**3
        der_end = a1 + 2*a2*L + 3*a3*L**2
        mean = a0 + a1*L/2 + a2*L**2/3 + a3*L**3/4
        return xt.Spline4(
            val_start=val_start,
            der_start=der_start,
            val_end=val_end,
            der_end=der_end,
            mean=mean,
        )

    p_bfe = p0.copy()
    p_sb = p0.copy()

    bfe_kwargs = dict(length=length, h=0, num_integration_steps=100)
    if knc is not None:
        bfe_kwargs['knc'] = knc
    if ksc is not None:
        bfe_kwargs['ksc'] = ksc
    bfe = xt.BFieldExpansion(**bfe_kwargs)
    line_bfe = xt.Line(elements=[bfe])
    line_bfe.particle_ref = p0.copy()
    line_bfe.configure_spin('auto')
    line_bfe.build_tracker(_context=test_context)
    line_bfe.track(p_bfe)

    sb_kwargs = dict(length=length, n_steps=1000)
    if knc is not None:
        sb_kwargs['by'] = tuple(hermite_from_poly([c * brho for c in coeffs], length) for coeffs in knc)
    if ksc is not None:
        sb_kwargs['bx'] = tuple(hermite_from_poly([c * brho for c in coeffs], length) for coeffs in ksc)

    sb = xt.SplineBoris(**sb_kwargs)
    line_sb = xt.Line(elements=[sb])
    line_sb.particle_ref = p0.copy()
    line_sb.configure_spin('auto')
    line_sb.build_tracker(_context=test_context)
    line_sb.track(p_sb)

    p_bfe.move(_context=xo.context_default)
    p_sb.move(_context=xo.context_default)

    assert p_bfe.state[0] == 1
    assert p_sb.state[0] == 1

    for coord in ('x', 'px', 'y', 'py', 'zeta', 'delta',
                  'spin_x', 'spin_y', 'spin_z'):
        xo.assert_allclose(getattr(p_bfe, coord), getattr(p_sb, coord),
                           rtol=0, atol=5e-8)

    spin_norm_out = np.sqrt(
        p_bfe.spin_x[0]**2 + p_bfe.spin_y[0]**2 + p_bfe.spin_z[0]**2)
    xo.assert_allclose(spin_norm_out, 1.0, rtol=0, atol=1e-10)
