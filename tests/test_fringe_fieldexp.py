"""Hard-edge fringe maps against BFieldExpansion magnets with smooth ramps
(see _fringe_fieldexp.py for the method and the acceptance criteria)."""

import _fringe_fieldexp as fe
import numpy as np
import pytest
from numpy.polynomial import Polynomial

import xtrack as xt

LENGTH = 0.5  # flat part of the field
ELLS = [0.04, 0.02, 0.01]


def _bfe(rows, ell, refine, steps_per_m):
    return xt.BFieldExpansion(
        length=ell,
        pkin_const=True,
        num_phi=6,
        num_integration_steps=max(40, int(steps_per_m * ell * refine)),
        **rows,
    )


def _multipole_reference(order, k, skew):
    key = 'ksc' if skew else 'knc'

    def rows(profile):
        return {key: fe.pure_multipole_rows(order, profile, skew)}

    def reference(ell, refine):
        return fe.make_tracker(
            [
                _bfe(rows(k * fe.ramp(ell, True)), ell, refine, 20000),
                _bfe(rows(Polynomial([k])), LENGTH, refine, 1000),
                _bfe(rows(k * fe.ramp(ell, False)), ell, refine, 20000),
            ]
        )

    return reference


def _hard_edge(magnet, ell):
    # Faces at the ramp midpoints: same integrated strength
    return fe.make_tracker([xt.Drift(length=ell / 2), magnet, xt.Drift(length=ell / 2)])


def _qsad_f1(ell):
    # Lee-Whiting: f1^2 / 24 = -int (r(s) - H(s - s0)) (s - s0) ds, with r the
    # ramp, H the hard edge and s0 = ell / 2 (f2 = 0 for a symmetric ramp)
    u = Polynomial([-0.5, 1])
    inner = (fe.SMOOTHSTEP * u).integ()
    outer = ((fe.SMOOTHSTEP - 1) * u).integ()
    moment = (inner(0.5) - inner(0)) + (outer(1) - outer(0.5))
    return np.sqrt(-24 * moment) * ell


MULTIPOLES = {
    # name: order, strength, skew, fit orders, kept order, ratio, min slope
    'quad': (1, 1.0, False, (1, 3, 5), [3], 0.03, 0.9),
    'skew_quad': (1, 1.0, True, (1, 3, 5), [3], 0.03, 0.9),
    # the k^2 terms of the body ends set a floor: ratio only
    'sextupole': (2, 20.0, False, (1, 2, 3, 4, 5, 6), [4], 0.25, None),
    # at ell = 0.04 the k^2 ell^2 terms of the body ends dominate
    'octupole': (3, 3000.0, False, (1, 3, 5, 7), [5], 0.4, 1.0, [0.01, 0.005, 0.0025]),
}


@pytest.mark.parametrize('name', MULTIPOLES)
def test_mult_fringe_vs_field_expansion(name):
    order, k, skew, orders, keep, ratio, min_slope, *ells = MULTIPOLES[name]

    def model(ell, fringe):
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

    rows = fe.scan(
        _multipole_reference(order, k, skew),
        model,
        ells[0] if ells else ELLS,
        orders=orders,
        keep=keep,
    )
    fe.report(name, rows)
    fe.check(rows, ratio=ratio, cap=0.05, min_slope=min_slope)


def test_qsad_fringe_vs_field_expansion():
    # The linear part: qsad corrects the O(ell^2) term of the hard edge
    def model(ell, qsad):
        qint1 = _qsad_f1(ell) if qsad else 0
        quad = xt.Quadrupole(
            length=LENGTH + ell,
            k1=1.0,
            edge_entry_active=True,
            edge_exit_active=True,
            edge_entry_qint1=qint1,
            edge_exit_qint1=qint1,
            integrator='yoshida-6',
            num_multipole_kicks=200,
        )
        return _hard_edge(quad, ell)

    rows = fe.scan(_multipole_reference(1, 1.0, False), model, ELLS, keep=[1])
    fe.report('qsad', rows)
    fe.check(rows, ratio=0.03, cap=1e-6, min_slope=2.8)


def test_bend_fringe_gap_vs_field_expansion():
    # Linear part of a dipole ramp in a straight frame, fint * hgap from the
    # ramp: fint * hgap = int r (1 - r) ds / 2
    k0 = 0.2

    def reference(ell, refine):
        def rows(profile):
            return {'knc': [list(profile.coef)]}

        return fe.make_tracker(
            [
                _bfe(rows(k0 * fe.ramp(ell, True)), ell, refine, 20000),
                _bfe(rows(Polynomial([k0])), LENGTH, refine, 1000),
                _bfe(rows(k0 * fe.ramp(ell, False)), ell, refine, 20000),
            ]
        )

    def model(ell, fringe):
        moment = (fe.SMOOTHSTEP * (1 - fe.SMOOTHSTEP)).integ()
        fint_hgap = (moment(1) - moment(0)) * ell / 2
        face = {
            f'edge_{ff}_{kk}': vv
            for ff in ('entry', 'exit')
            for kk, vv in dict(
                model='dipole-only', active=fringe, fint=0.5, hgap=2 * fint_hgap
            ).items()
        }
        bend = xt.Bend(
            length=LENGTH + ell,
            k0=k0,
            angle=0,
            **face,
            # Exact dipole map: no kicks needed for a pure dipole
            model='bend-kick-bend',
            integrator='uniform',
            num_multipole_kicks=1,
        )
        return _hard_edge(bend, ell)

    rows = fe.scan(reference, model, ELLS, orders=(0, 1, 2, 3, 4, 5), keep=[1])
    fe.report('bend gap', rows)
    fe.check(rows, ratio=0.003, cap=1e-4, min_slope=0.9)
