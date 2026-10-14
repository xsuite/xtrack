import numpy as np
import xobjects as xo
from xobjects.test_helpers import for_all_test_contexts

import xtrack as xt

N_KICKS_YOSHIDA = 7


def make_particles(context, off_momentum=False):
    if off_momentum:
        return xt.Particles(
            p0c=7e12, x=2e-2, px=-3e-3, y=-1.5e-2, py=2e-3, delta=0.1,
            _context=context,
        )
    return xt.Particles(
        p0c=7e12, x=2e-7, px=-3e-8, y=-1.5e-7, py=2e-8, delta=0,
        _context=context,
    )


def max_error(p_test, p_ref, coordinates):
    return max(
        np.max(
            np.abs(
                np.asarray(getattr(p_test, coordinate))
                - np.asarray(getattr(p_ref, coordinate))
            )
        )
        for coordinate in coordinates
    )


@for_all_test_contexts
def test_mat_kick_mat_exact_basic_correctness(test_context):
    # A single-slice mat-kick-mat-exact quadrupole should match a converged
    # reference to near machine precision.
    quad = xt.Quadrupole(
        length=1.2,
        k1=0.37,
        model='mat-kick-mat-exact',
        num_multipole_kicks=1,
        _context=test_context,
    )
    reference = quad.copy()
    reference.model = 'drift-kick-drift-exact'
    reference.num_multipole_kicks = 128

    p0 = make_particles(test_context)
    p_test = p0.copy()
    p_ref = p0.copy()

    quad.track(p_test)
    reference.track(p_ref)

    for coordinate in ('x', 'px', 'y', 'py', 'zeta'):
        test_value = getattr(p_test, coordinate)
        assert np.all(np.isfinite(test_value))
        xo.assert_allclose(test_value, getattr(p_ref, coordinate), rtol=0, atol=5e-13)


@for_all_test_contexts
def test_mat_kick_mat_exact_converges_off_momentum(test_context):
    # Off-momentum, with a dipole kick (knl[0]) on top of k1: mat-kick-mat-exact
    # should converge far faster than plain mat-kick-mat.
    num_slices = 10
    coordinates = ('x', 'px', 'y', 'py', 'zeta')

    quad = xt.Quadrupole(
        length=1.2,
        k1=0.37,
        knl=[0.05 * 1.2],
        integrator='yoshida-6',
        _context=test_context,
    )

    p0 = make_particles(test_context, off_momentum=True)

    reference = quad.copy()
    reference.model = 'drift-kick-drift-exact'
    reference.num_multipole_kicks = 15 * 400
    p_ref = p0.copy()
    reference.track(p_ref)

    exact = quad.copy()
    exact.model = 'mat-kick-mat-exact'
    exact.num_multipole_kicks = num_slices * N_KICKS_YOSHIDA
    p_exact = p0.copy()
    exact.track(p_exact)

    plain = quad.copy()
    plain.model = 'mat-kick-mat'
    plain.num_multipole_kicks = num_slices * N_KICKS_YOSHIDA
    p_plain = p0.copy()
    plain.track(p_plain)

    exact_error = max_error(p_exact, p_ref, coordinates)
    plain_error = max_error(p_plain, p_ref, coordinates)

    assert exact_error < 1e-12
    assert exact_error < 1e-3 * plain_error


@for_all_test_contexts
def test_mat_kick_mat_exact_empty_kick(test_context):
    # k1 = 0 and no multipoles: the kick is empty and the element is a drift.
    # The path correction must still be applied.
    quad = xt.Quadrupole(
        length=1.2, k1=0.0, model='mat-kick-mat-exact', num_multipole_kicks=3,
        _context=test_context,
    )
    reference = quad.copy()
    reference.model = 'drift-kick-drift-exact'

    p0 = make_particles(test_context, off_momentum=True)
    p_test = p0.copy()
    p_ref = p0.copy()
    quad.track(p_test)
    reference.track(p_ref)

    for coordinate in ('x', 'px', 'y', 'py', 'zeta'):
        xo.assert_allclose(
            getattr(p_test, coordinate), getattr(p_ref, coordinate),
            rtol=0, atol=1e-14)


def test_mat_kick_mat_exact_available_models():
    assert 'mat-kick-mat-exact' in xt.Quadrupole.get_available_models()
    assert 'mat-kick-mat-exact' in xt.Multipole.get_available_models()
    assert 'mat-kick-mat-exact' not in xt.Magnet.get_available_models()
    assert 'mat-kick-mat-exact' not in xt.Bend.get_available_models()
    assert 'mat-kick-mat-exact' not in xt.Cavity.get_available_models()
    assert 'mat-kick-mat-exact' not in xt.CrabCavity.get_available_models()

    quad = xt.Quadrupole(model='mat-kick-mat-exact')
    assert quad.model == 'mat-kick-mat-exact'
