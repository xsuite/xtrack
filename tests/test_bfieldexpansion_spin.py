# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

import numpy as np
import pytest
import xobjects as xo
import xtrack as xt
from xobjects.test_helpers import for_all_test_contexts


@for_all_test_contexts
@pytest.mark.parametrize("h, k0", [(0.1, 0.1),  (0.1, 0), (0, 0.1), (-0.1, -0.1)])  # Test curved bend, curved drift, straight bend, negative curvature
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
        knc=np.array([[k0]]),
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


@for_all_test_contexts
def test_bfieldexpansion_spin_vs_splineboris(test_context):
    length = 1.0
    knc = np.array([[0.2, 0.01, -0.005], [0.10, -0.04, 0.02]])
    ksc = np.array([[0.015, -0.008, 0.004], [0.08, 0.03, -0.015]])

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
        spin_z=0.3
    )

    brho = p0.rigidity0[0]

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

    bfe = xt.BFieldExpansion(knc=knc, ksc=ksc, length=length, h=0, num_integration_steps=100)
    line_bfe = xt.Line(elements=[bfe])
    line_bfe.particle_ref = p0.copy()
    line_bfe.configure_spin('auto')
    line_bfe.build_tracker(_context=test_context)
    line_bfe.track(p_bfe)

    sb = xt.SplineBoris(length=length, n_steps=1000, 
                        by=tuple(hermite_from_poly([c * brho for c in coeffs], length) for coeffs in knc), 
                        bx=tuple(hermite_from_poly([c * brho for c in coeffs], length) for coeffs in ksc))
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