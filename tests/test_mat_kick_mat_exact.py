import numpy as np
import xobjects as xo
from xobjects.test_helpers import for_all_test_contexts

import xtrack as xt

_TRANSVERSE = ('x', 'px', 'y', 'py')
_COMPILED_CONTEXTS = set()


def _particle(context, off_momentum=False):
    if off_momentum:
        return xt.Particles(
            p0c=7e12, x=2e-2, px=-3e-3, y=-1.5e-2, py=2e-3, delta=0.1,
            _context=context,
        )
    return xt.Particles(
        p0c=7e12, x=2e-7, px=-3e-8, y=-1.5e-7, py=2e-8, delta=0,
        _context=context,
    )


def _track_magnet(context, model, num_kicks, k0=0.0, k1=0.37, angle=0.0,
                   off_momentum=False, integrator='uniform'):
    magnet = xt.Magnet(
        length=1.2,
        k0=k0,
        k1=k1,
        angle=angle,
        model=model,
        integrator=integrator,
        num_multipole_kicks=num_kicks,
        edge_entry_active=False,
        edge_exit_active=False,
        _context=context,
    )
    context_id = id(context)
    if context_id not in _COMPILED_CONTEXTS:
        magnet.compile_kernels(only_if_needed=False)
        _COMPILED_CONTEXTS.add(context_id)
    particle = _particle(context, off_momentum=off_momentum)
    magnet.track(particle)
    return particle


def _max_error(particle, reference, coordinates):
    return max(
        np.max(
            np.abs(
                np.asarray(getattr(particle, coordinate))
                - np.asarray(getattr(reference, coordinate))
            )
        )
        for coordinate in coordinates
    )


@for_all_test_contexts
def test_mat_kick_mat_exact_basic_correctness(test_context):
    # Pure quadrupole (k0=0): body map is momentum-linear and single-kick
    # mat-kick-mat-exact should agree with a converged reference to near
    # machine precision.
    assert 'mat-kick-mat-exact' in xt.Magnet.get_available_models()
    exact = _track_magnet(test_context, 'mat-kick-mat-exact', 1, k0=0.0, k1=0.37)
    dkd = _track_magnet(test_context, 'drift-kick-drift-exact', 128, k0=0.0, k1=0.37)

    for coordinate in _TRANSVERSE:
        exact_value = np.asarray(getattr(exact, coordinate))
        dkd_value = np.asarray(getattr(dkd, coordinate))
        assert np.all(np.isfinite(exact_value))
        xo.assert_allclose(exact_value, dkd_value, rtol=0, atol=5e-13)


@for_all_test_contexts
def test_mat_kick_mat_exact_converges_off_momentum(test_context):
    # Straight (h=0) combined k0/k1 magnet: mat-kick-mat-exact should need far
    # fewer kicks than plain mat-kick-mat to approach the converged reference.
    kwargs = dict(k0=0.05, k1=0.37, off_momentum=True, integrator='yoshida4')
    reference = _track_magnet(
        test_context, 'drift-kick-drift-exact', 15 * 400,
        integrator='yoshida4', k0=0.05, k1=0.37, off_momentum=True)
    coordinates = _TRANSVERSE

    exact_error = _max_error(
        _track_magnet(test_context, 'mat-kick-mat-exact', 7, **kwargs),
        reference, coordinates)
    plain_error = _max_error(
        _track_magnet(test_context, 'mat-kick-mat', 7, **kwargs),
        reference, coordinates)

    assert exact_error < 1e-3 * plain_error


@for_all_test_contexts
def test_mat_kick_mat_exact_k0_alone_gets_correction(test_context):
    # k1 == 0, h == 0: a pure dipole kick. The correction must still apply
    # (i.e. must not be silently skipped by the "kick is empty" heuristic).
    kwargs = dict(k0=0.05, k1=0.0, off_momentum=True, integrator='yoshida4')
    reference = _track_magnet(
        test_context, 'drift-kick-drift-exact', 15 * 400,
        integrator='yoshida4', k0=0.05, k1=0.0, off_momentum=True)

    exact_error = _max_error(
        _track_magnet(test_context, 'mat-kick-mat-exact', 7, **kwargs),
        reference, _TRANSVERSE)
    plain_error = _max_error(
        _track_magnet(test_context, 'mat-kick-mat', 7, **kwargs),
        reference, _TRANSVERSE)

    assert exact_error < plain_error


@for_all_test_contexts
def test_mat_kick_mat_exact_matches_plain_when_curved(test_context):
    # h != 0: the correction is not exact for a curved reference frame, so
    # mat-kick-mat-exact must fall back to being identical to mat-kick-mat.
    kwargs = dict(k0=0.02, k1=0.37, angle=0.02, off_momentum=True)
    exact = _track_magnet(test_context, 'mat-kick-mat-exact', 5, **kwargs)
    plain = _track_magnet(test_context, 'mat-kick-mat', 5, **kwargs)

    for coordinate in _TRANSVERSE + ('zeta', 'delta'):
        xo.assert_allclose(
            np.asarray(getattr(exact, coordinate)),
            np.asarray(getattr(plain, coordinate)),
            rtol=0, atol=0,
        )


def test_mat_kick_mat_exact_model_roundtrip():
    assert 'mat-kick-mat-exact' in xt.Magnet.get_available_models()
    assert 'mat-kick-mat-exact' in xt.Quadrupole.get_available_models()
    assert 'mat-kick-mat-exact' not in xt.Cavity.get_available_models()
    assert 'mat-kick-mat-exact' not in xt.CrabCavity.get_available_models()

    magnet = xt.Magnet(model='mat-kick-mat-exact')
    assert magnet.model == 'mat-kick-mat-exact'
