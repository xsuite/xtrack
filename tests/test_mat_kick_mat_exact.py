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
    # Pure quadrupole (k0=0): a single-slice mat-kick-mat-exact should match
    # a converged reference to near machine precision.
    assert 'mat-kick-mat-exact' in xt.Magnet.get_available_models()

    magnet = xt.Magnet(
        length=1.2,
        k0=0.0,
        k1=0.37,
        model='mat-kick-mat-exact',
        num_multipole_kicks=1,
        edge_entry_active=False,
        edge_exit_active=False,
        _context=test_context,
    )
    reference = magnet.copy()
    reference.model = 'drift-kick-drift-exact'
    reference.num_multipole_kicks = 128

    p0 = make_particles(test_context)
    p_test = p0.copy()
    p_ref = p0.copy()

    magnet.track(p_test)
    reference.track(p_ref)

    for coordinate in ('x', 'px', 'y', 'py', 'zeta'):
        test_value = getattr(p_test, coordinate)
        assert np.all(np.isfinite(test_value))
        xo.assert_allclose(test_value, getattr(p_ref, coordinate), rtol=0, atol=5e-13)


@for_all_test_contexts
def test_mat_kick_mat_exact_converges_off_momentum(test_context):
    # Straight (h=0) combined k0/k1 magnet: mat-kick-mat-exact should converge
    # far faster than plain mat-kick-mat, and reach 1e-14 with enough slices.
    num_slices = 10
    coordinates = ('x', 'px', 'y', 'py', 'zeta')

    magnet = xt.Magnet(
        length=1.2,
        k0=0.05,
        k1=0.37,
        integrator='yoshida4',
        edge_entry_active=False,
        edge_exit_active=False,
        _context=test_context,
    )

    p0 = make_particles(test_context, off_momentum=True)

    reference = magnet.copy()
    reference.model = 'drift-kick-drift-exact'
    reference.num_multipole_kicks = 15 * 400
    p_ref = p0.copy()
    reference.track(p_ref)

    exact = magnet.copy()
    exact.model = 'mat-kick-mat-exact'
    exact.num_multipole_kicks = num_slices * N_KICKS_YOSHIDA
    p_exact = p0.copy()
    exact.track(p_exact)

    plain = magnet.copy()
    plain.model = 'mat-kick-mat'
    plain.num_multipole_kicks = num_slices * N_KICKS_YOSHIDA
    p_plain = p0.copy()
    plain.track(p_plain)

    exact_error = max_error(p_exact, p_ref, coordinates)
    plain_error = max_error(p_plain, p_ref, coordinates)

    assert exact_error < 1e-14
    assert exact_error < 1e-3 * plain_error


@for_all_test_contexts
def test_mat_kick_mat_exact_k0_alone_gets_correction(test_context):
    # k1 == 0, h == 0: pure dipole kick, Kx = k0*h + k1 == 0. The correction
    # must still apply (not be silently skipped as an "empty kick").
    num_slices = 10
    transverse = ('x', 'px', 'y', 'py')

    magnet = xt.Magnet(
        length=1.2,
        k0=0.05,
        k1=0.0,
        integrator='yoshida4',
        edge_entry_active=False,
        edge_exit_active=False,
        _context=test_context,
    )

    p0 = make_particles(test_context, off_momentum=True)

    reference = magnet.copy()
    reference.model = 'drift-kick-drift-exact'
    reference.num_multipole_kicks = 15 * 400
    p_ref = p0.copy()
    reference.track(p_ref)

    exact = magnet.copy()
    exact.model = 'mat-kick-mat-exact'
    exact.num_multipole_kicks = num_slices * N_KICKS_YOSHIDA
    p_exact = p0.copy()
    exact.track(p_exact)

    plain = magnet.copy()
    plain.model = 'mat-kick-mat'
    plain.num_multipole_kicks = num_slices * N_KICKS_YOSHIDA
    p_plain = p0.copy()
    plain.track(p_plain)

    exact_error = max_error(p_exact, p_ref, transverse)
    plain_error = max_error(p_plain, p_ref, transverse)

    assert exact_error < 1e-14
    assert exact_error < plain_error

    num_slices_zeta = 1000

    exact_zeta = magnet.copy()
    exact_zeta.model = 'mat-kick-mat-exact'
    exact_zeta.num_multipole_kicks = num_slices_zeta * N_KICKS_YOSHIDA
    p_exact_zeta = p0.copy()
    exact_zeta.track(p_exact_zeta)

    plain_zeta = magnet.copy()
    plain_zeta.model = 'mat-kick-mat'
    plain_zeta.num_multipole_kicks = num_slices_zeta * N_KICKS_YOSHIDA
    p_plain_zeta = p0.copy()
    plain_zeta.track(p_plain_zeta)

    exact_zeta_error = max_error(p_exact_zeta, p_ref, ('zeta',))
    plain_zeta_error = max_error(p_plain_zeta, p_ref, ('zeta',))

    assert exact_zeta_error < plain_zeta_error


@for_all_test_contexts
def test_mat_kick_mat_exact_matches_plain_when_curved(test_context):
    # h != 0: correction isn't exact in a curved frame, so mat-kick-mat-exact
    # must fall back to being identical to mat-kick-mat.
    magnet = xt.Magnet(
        length=1.2,
        k0=0.02,
        k1=0.37,
        angle=0.02,
        num_multipole_kicks=5,
        edge_entry_active=False,
        edge_exit_active=False,
        _context=test_context,
    )

    p0 = make_particles(test_context, off_momentum=True)

    exact = magnet.copy()
    exact.model = 'mat-kick-mat-exact'
    p_exact = p0.copy()
    exact.track(p_exact)

    plain = magnet.copy()
    plain.model = 'mat-kick-mat'
    p_plain = p0.copy()
    plain.track(p_plain)

    for coordinate in ('x', 'px', 'y', 'py', 'zeta', 'delta'):
        xo.assert_allclose(
            getattr(p_exact, coordinate),
            getattr(p_plain, coordinate),
            rtol=0,
            atol=0,
        )


def test_mat_kick_mat_exact_model_roundtrip():
    assert 'mat-kick-mat-exact' in xt.Magnet.get_available_models()
    assert 'mat-kick-mat-exact' in xt.Quadrupole.get_available_models()
    assert 'mat-kick-mat-exact' not in xt.Cavity.get_available_models()
    assert 'mat-kick-mat-exact' not in xt.CrabCavity.get_available_models()

    magnet = xt.Magnet(model='mat-kick-mat-exact')
    assert magnet.model == 'mat-kick-mat-exact'
