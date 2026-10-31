"""Hard-edge fringe maps against BFieldExpansion magnets with smooth ramps
(method in _fringe_fieldexp.py)."""

import _fringe_fieldexp as fe
import numpy as np
import pytest
from numpy.polynomial import Polynomial

import xtrack as xt

LENGTH = 0.5  # flat part of the field
ELLS = [0.04, 0.02, 0.01]


def _bfe(rows, ell, steps_per_m, **kwargs):
    return xt.BFieldExpansion(
        length=ell,
        num_integration_steps=max(40, int(steps_per_m * ell)),
        **{'pkin_const': True, 'num_phi': 6, **kwargs},
        **rows,
    )


def _reference(rows, gauge_ks=0, **kwargs):
    """rows(shape, k): field rows of strength k shape(s), shape from 0 to 1.
    gauge_ks: compare in the symmetric gauge of a solenoid ks. kwargs: of
    BFieldExpansion."""

    def build(ell):
        def line(k):
            elements = [
                _bfe(rows(fe.ramp(ell, True), k), ell, 20000, **kwargs),
                _bfe(rows(Polynomial([1.]), k), LENGTH, 1000, **kwargs),
                _bfe(rows(fe.ramp(ell, False), k), ell, 20000, **kwargs),
            ]
            if gauge_ks:
                elements = [fe.landau_to_symmetric(-gauge_ks), *elements,
                            fe.landau_to_symmetric(gauge_ks)]
            return fe.make_line(elements)

        return line

    return build


def _hard_edge(magnet, ell):
    # Faces at the ramp midpoints: same integrated strength
    return fe.make_line(
        [xt.DriftExact(length=ell / 2), magnet, xt.DriftExact(length=ell / 2)]
    )


def _check(name, reference, model, power, degree, rate=None, gain=1e-4, tol=0.1):
    """rate: power of ell of the residual (within tol); None: exact. gain: the
    residual with the fringe, relative to the error without it."""
    errs = []
    for ell in ELLS:
        ref = fe.strength_part(reference(ell), power, degree)
        errs.append(fe.rel_diff(fe.strength_part(model(ell, True), power, degree), ref))
    none = fe.rel_diff(fe.strength_part(model(ELLS[-1], False), power, degree), ref)
    print(f'{name}: without {none:.1e}, with', ', '.join(f'{ee:.1e}' for ee in errs))
    assert errs[-1] < gain * none
    if rate is None:
        assert max(errs) < 1e-13
    else:
        assert abs(fe.slope(ELLS, errs) - rate) < tol


def _quad_lw_f1(ell):
    # Lee-Whiting: f1^2 / 24 = -int (r(s) - H(s - s0)) (s - s0) ds, with r the
    # ramp, H the hard edge and s0 = ell / 2 (f2 = 0 for a symmetric ramp)
    u = Polynomial([-0.5, 1])
    inner = (fe.SMOOTHSTEP * u).integ()
    outer = ((fe.SMOOTHSTEP - 1) * u).integ()
    moment = (inner(0.5) - inner(0)) + (outer(1) - outer(0.5))
    return np.sqrt(-24 * moment) * ell


def _fint_hgap(ell):
    # Soft dipole fringe: fint * hgap = int r (1 - r) ds / 2
    moment = (fe.SMOOTHSTEP * (1 - fe.SMOOTHSTEP)).integ()
    return (moment(1) - moment(0)) * ell / 2


MULTIPOLES = {
    # name: order, skew; fringe term: degree order + 2, linear in k
    'quad': (1, False),
    'skew_quad': (1, True),
    'sextupole': (2, False),
    'octupole': (3, False),
}


@pytest.mark.parametrize('name', MULTIPOLES)
def test_mult_fringe_vs_field_expansion(name):
    order, skew = MULTIPOLES[name]
    key = 'ksc' if skew else 'knc'
    reference = _reference(
        lambda shape, k: {key: fe.pure_multipole_rows(order, k * shape, skew)})

    def model(ell, fringe):
        def build(k):
            kl = np.zeros(order + 1)
            kl[order] = k * (LENGTH + ell)
            magnet = xt.Magnet(
                length=LENGTH + ell,
                edge_entry_model='full',
                edge_exit_model='full',
                edge_entry_active=fringe,
                edge_exit_active=fringe,
                integrator='yoshida-6',
                num_multipole_kicks=200,
                **({'ksl': kl} if skew else {'knl': kl}),
            )
            return _hard_edge(magnet, ell)

        return build

    _check(name, reference, model, power=1, degree=order + 2, rate=2)


def test_quad_lw_fringe_vs_field_expansion():
    # Linear map, linear in k1: set by the moments 0, 1, 2 of k1(s), which the
    # hard edge and f1 reproduce, so exact
    def model(ell, quad_lw):
        qint1 = _quad_lw_f1(ell) if quad_lw else 0

        def build(k):
            quad = xt.Quadrupole(
                length=LENGTH + ell,
                k1=k,
                edge_entry_model='full',
                edge_exit_model='full',
                edge_entry_qint1=qint1,
                edge_exit_qint1=qint1,
                model='mat-kick-mat-exact',
                integrator='yoshida-6',
                num_multipole_kicks=1,
            )
            return _hard_edge(quad, ell)

        return build

    reference = _reference(lambda shape, k: {'knc': fe.pure_multipole_rows(1, k * shape)})
    _check('quad_lw', reference, model, power=1, degree=1)


def test_bend_fringe_gap_vs_field_expansion():
    # Linear map, quadratic in k0
    def model(ell, fringe):
        face = {
            f'edge_{ff}_{kk}': vv
            for ff in ('entry', 'exit')
            for kk, vv in {
                'model': 'no-mult', 'active': fringe, 'fint': 0.5, 'hgap': 2 * _fint_hgap(ell)
            }.items()
        }

        def build(k):
            bend = xt.Bend(
                length=LENGTH + ell,
                k0=k,
                angle=0,
                **face,
                # Exact dipole map: no kicks needed for a pure dipole
                model='bend-kick-bend',
                integrator='uniform',
                num_multipole_kicks=1,
            )
            return _hard_edge(bend, ell)

        return build

    reference = _reference(lambda shape, k: {'knc': [list((k * shape).coef)]})
    _check('bend gap', reference, model, power=2, degree=1, rate=2)


def test_quad_lw_fringe_in_solenoid_vs_field_expansion():
    # A quadrupole face in a uniform solenoid field: Lee-Whiting on the kinetic
    # momenta (SAD tqlfre, bz != 0). Model: variable solenoids with a
    # hard-edge k1. Linear map, linear in k1, in the symmetric gauge of the
    # solenoid faces: the residual falls as ell^4
    ks = 0.5

    def model(ell, quad_lw):
        qint1 = _quad_lw_f1(ell) if quad_lw else 0

        def build(k):
            def solenoid(length, **kwargs):
                return xt.VariableSolenoid(length=length, ks_profile=[ks, ks], **kwargs)

            body = solenoid(
                LENGTH + ell, knl=[0, k * (LENGTH + ell)], integrator='yoshida-6',
                num_multipole_kicks=50, edge_entry_model='no-mult',
                edge_exit_model='no-mult', edge_entry_qint1=qint1,
                edge_exit_qint1=qint1)
            return fe.make_line([solenoid(ell / 2), body, solenoid(ell / 2)])

        return build

    reference = _reference(
        lambda shape, k: {'ksolc': [ks], 'knc': fe.pure_multipole_rows(1, k * shape)},
        gauge_ks=ks, pkin_const=False)
    _check('quad_lw in solenoid', reference, model, power=1, degree=1, rate=4, tol=0.2)


def test_combined_face_vs_field_expansion():
    # A combined-function face: k0 = h (the orbit stays on the axis) and k1
    # ramp together on the bent field expansion; the soft dipole fringe and
    # Lee-Whiting from the moments of the ramp. Linear map, linear in k1: the
    # residual falls as ell^2, the two fringes have no cross terms. Rows 0 and
    # 1 only (with h != 0 the x^3 row does not converge on a short ramp).
    h = 0.05

    def model(ell, fringe):
        soft = {'fint': 0.5, 'hgap': 2 * _fint_hgap(ell), 'qint1': _quad_lw_f1(ell)}
        face = {
            f'edge_{ff}_{kk}': vv
            for ff in ('entry', 'exit')
            for kk, vv in {'model': 'full', 'active': fringe, **soft}.items()
        }

        def build(k):
            def drift(length):
                return xt.Bend(length=length, k0=0, angle=h * length,
                               edge_entry_model='suppressed',
                               edge_exit_model='suppressed')

            bend = xt.Bend(length=LENGTH + ell, angle=h * (LENGTH + ell), k0=h,
                           k1=k, **face, integrator='yoshida-6',
                           num_multipole_kicks=200)
            return fe.make_line([drift(ell / 2), bend, drift(ell / 2)])

        return build

    reference = _reference(
        lambda shape, k: {'knc': [list((h * shape).coef), list((k * shape).coef)]},
        h=h, num_phi=8)
    _check('combined face', reference, model, power=1, degree=1, rate=2, gain=0.05)
