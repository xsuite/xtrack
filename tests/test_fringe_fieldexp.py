"""Hard-edge fringe maps against BFieldExpansion magnets with smooth ramps
(method in _fringe_fieldexp.py)."""

import _fringe_fieldexp as fe
import numpy as np
import pytest
from numpy.polynomial import Polynomial

import xtrack as xt

LENGTH = 0.5  # flat part of the field
ELLS = [0.04, 0.02, 0.01]


def _bfe(rows, ell, steps_per_m):
    return xt.BFieldExpansion(
        length=ell,
        pkin_const=True,
        num_phi=6,
        num_integration_steps=max(40, int(steps_per_m * ell)),
        **rows,
    )


def _reference(rows):
    """rows(profile): field rows of strength profile(s)."""

    def build(ell):
        return lambda k: fe.make_line(
            [
                _bfe(rows(k * fe.ramp(ell, True)), ell, 20000),
                _bfe(rows(Polynomial([k])), LENGTH, 1000),
                _bfe(rows(k * fe.ramp(ell, False)), ell, 20000),
            ]
        )

    return build


def _hard_edge(magnet, ell):
    # Faces at the ramp midpoints: same integrated strength
    return fe.make_line(
        [xt.DriftExact(length=ell / 2), magnet, xt.DriftExact(length=ell / 2)]
    )


def _check(name, reference, model, power, degree, rate=None):
    """rate: power of ell of the residual; None: exact."""
    errs = []
    for ell in ELLS:
        ref = fe.strength_part(reference(ell), power, degree)
        errs.append(fe.rel_diff(fe.strength_part(model(ell, True), power, degree), ref))
    none = fe.rel_diff(fe.strength_part(model(ELLS[-1], False), power, degree), ref)
    print(f'{name}: without {none:.1e}, with', ', '.join(f'{ee:.1e}' for ee in errs))
    assert errs[-1] < 1e-4 * none
    if rate is None:
        assert max(errs) < 1e-13
    else:
        assert abs(fe.slope(ELLS, errs) - rate) < 0.1


def _qsad_f1(ell):
    # Lee-Whiting: f1^2 / 24 = -int (r(s) - H(s - s0)) (s - s0) ds, with r the
    # ramp, H the hard edge and s0 = ell / 2 (f2 = 0 for a symmetric ramp)
    u = Polynomial([-0.5, 1])
    inner = (fe.SMOOTHSTEP * u).integ()
    outer = ((fe.SMOOTHSTEP - 1) * u).integ()
    moment = (inner(0.5) - inner(0)) + (outer(1) - outer(0.5))
    return np.sqrt(-24 * moment) * ell


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
    reference = _reference(lambda p: {key: fe.pure_multipole_rows(order, p, skew)})

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


def test_qsad_fringe_vs_field_expansion():
    # Linear map, linear in k1: set by the moments 0, 1, 2 of k1(s), which the
    # hard edge and f1 reproduce, so exact
    def model(ell, qsad):
        qint1 = _qsad_f1(ell) if qsad else 0

        def build(k):
            quad = xt.Quadrupole(
                length=LENGTH + ell,
                k1=k,
                edge_entry_active=True,
                edge_exit_active=True,
                edge_entry_qint1=qint1,
                edge_exit_qint1=qint1,
                model='mat-kick-mat-exact',
                integrator='yoshida-6',
                num_multipole_kicks=1,
            )
            return _hard_edge(quad, ell)

        return build

    reference = _reference(lambda p: {'knc': fe.pure_multipole_rows(1, p)})
    _check('qsad', reference, model, power=1, degree=1)


def test_bend_fringe_gap_vs_field_expansion():
    # Linear map, quadratic in k0; fint * hgap = int r (1 - r) ds / 2
    def model(ell, fringe):
        moment = (fe.SMOOTHSTEP * (1 - fe.SMOOTHSTEP)).integ()
        fint_hgap = (moment(1) - moment(0)) * ell / 2
        face = {
            f'edge_{ff}_{kk}': vv
            for ff in ('entry', 'exit')
            for kk, vv in dict(
                model='no-mult', active=fringe, fint=0.5, hgap=2 * fint_hgap
            ).items()
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

    reference = _reference(lambda p: {'knc': [list(p.coef)]})
    _check('bend gap', reference, model, power=2, degree=1, rate=2)
