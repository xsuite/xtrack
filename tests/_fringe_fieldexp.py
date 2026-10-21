"""Comparison of hard-edge fringe maps with a field of smooth ramps.

The reference is a magnet whose strength rises from 0 over a ramp of length
ell (a C2 smoothstep), integrated through its field expansion; the model is
the hard-edge magnet with its faces at the ramp midpoints (same integrated
strength) and the fringe map under test. Only the nonlinear part of the maps
is compared: the linear part of a fringe of width ell differs by O(ell) for
any hard-edge model.

A fringe map is accepted when, at every ell of a scan,
  - the reference is converged (step refinement) well below the error,
  - err(with fringe) < ratio * err(without fringe),
  - err(with fringe) < cap * |nonlinear part of the reference|,
and err(with fringe) falls with ell at least at the expected rate.
The fringe parameters are mapped from the ramp analytically, never fitted.
"""
from math import factorial

import numpy as np
from numpy.polynomial import Polynomial

import xtrack as xt

SMOOTHSTEP = Polynomial([0, 0, 0, 10, -15, 6])

# Starting amplitudes (x, px, y, py); scaled by 1, 2, ... in the fit
AMPLITUDES = 0.1 * np.array([[4e-3, 1e-5, 2e-3, -3e-5],
                             [3e-3, 1e-4, -3e-3, -2e-4],
                             [-2e-3, 2e-4, 4e-3, 1e-4]])


def ramp(ell, up=True):
    """Profile on [0, ell] going from 0 to 1 (up) or from 1 to 0."""
    u = Polynomial([0, 1 / ell]) if up else Polynomial([1, -1 / ell])
    return SMOOTHSTEP(u)


def pure_multipole_rows(order, profile, skew=False):
    """knc (ksc) rows of a pure 2(order+1)-pole whose strength k_order(s) is
    the polynomial ``profile``.

    BFieldExpansion fixes the midplane field: row j adds k x^j / j! to
    By(x, 0, s) (Bx for ksc). A single cylindrical harmonic m = order + 1,
    which the hard-edge fringe maps assume, has on the midplane
      By = sum_p (-1)^p m / (4^p p! (m+p)!) k^(2p) x^(order+2p)        normal
      Bx = sum_p (-1)^p (m+2p) / (4^p p! (m+p)!) k^(2p) x^(order+2p)   skew
    so it needs the rows order + 2p for p > 0 as well (for a quadrupole,
    the row 3 is -k''/2).
    """
    m = order + 1
    rows = {}
    for p in range(profile.degree() // 2 + 1):
        num = (m + 2 * p) if skew else m
        coeff = ((-1)**p * num / (4**p * factorial(p) * factorial(m + p))
                 * factorial(order + 2 * p))
        rows[order + 2 * p] = list(coeff * profile.deriv(2 * p).coef)
    return [rows.get(jj, [0.]) for jj in range(max(rows) + 1)]


def multipole_ramp(order, k, ell, up=True, skew=False, steps=400):
    """BFieldExpansion of a pure normal (skew) 2(order+1)-pole ramp of
    strength k, s in [0, ell]."""
    field = pure_multipole_rows(order, k * ramp(ell, up), skew)
    kw = dict(ksc=field) if skew else dict(knc=field)
    return xt.BFieldExpansion(length=ell, num_integration_steps=steps, **kw)


def make_tracker(elements, p0c=1e9):
    line = xt.Line(elements=elements)
    line.set_particle_ref('proton', p0c=p0c)
    line.build_tracker()

    def track(aa):
        p = line.build_particles(x=aa[:, 0], px=aa[:, 1],
                                 y=aa[:, 2], py=aa[:, 3])
        line.track(p)
        assert np.all(p.state == 1)
        return np.array([p.x, p.px, p.y, p.py]).T
    return track


def nonlinear_part(track, amplitudes=AMPLITUDES, orders=(1, 3, 5), keep=None):
    """Fitted terms of f(k a) = sum_n c_n k^n a^n at k = 1. Returns the sum of
    the orders in ``keep``; by default all except the linear one and the
    highest one (which takes the truncation error)."""
    kk = np.arange(1, len(orders) + 1, dtype=float)
    out = [track(k * amplitudes) for k in kk]
    mat = np.array([[k**n for n in orders] for k in kk])
    coef = np.linalg.solve(mat, np.array([oo.ravel() for oo in out]))
    if keep is None:
        keep = [n for n in orders if n != 1][:-1]
    idx = [orders.index(n) for n in keep]
    return coef[idx].sum(axis=0).reshape(out[0].shape)


def scan(reference, model, ells, orders=(1, 3, 5), keep=None):
    """reference(ell, steps) and model(ell, fringe: bool) return a track
    function; steps is a refinement factor. Returns one row per ell."""
    def part(track):
        return nonlinear_part(track, orders=orders, keep=keep)
    rows = []
    for ell in ells:
        ref = part(reference(ell, 1))
        ref2 = part(reference(ell, 2))
        with_f = part(model(ell, True))
        without = part(model(ell, False))
        rows.append(dict(
            ell=ell, scale=np.max(np.abs(ref)),
            conv=np.max(np.abs(ref2 - ref)),
            err_fringe=np.max(np.abs(with_f - ref)),
            err_none=np.max(np.abs(without - ref))))
    return rows


def report(name, rows):
    print(f'\n{name}')
    print(f'{"ell":>8} {"|ref|":>10} {"conv":>10} {"fringe":>10} {"none":>10}')
    for r in rows:
        print(f'{r["ell"]:8.4f} {r["scale"]:10.3e} {r["conv"]:10.1e} '
              f'{r["err_fringe"]:10.3e} {r["err_none"]:10.3e}')
    if len(rows) > 1:
        print(f'slope of err(fringe) in ell: {slope(rows):.2f}')


def slope(rows):
    ell = np.log([r['ell'] for r in rows])
    err = np.log([r['err_fringe'] for r in rows])
    return np.polyfit(ell, err, 1)[0]


def check(rows, ratio, cap, min_slope=None, conv_factor=0.1):
    for r in rows:
        assert r['conv'] < conv_factor * r['err_fringe'], r
        assert r['err_fringe'] < ratio * r['err_none'], r
        assert r['err_fringe'] < cap * r['scale'], r
    if min_slope is not None:
        assert slope(rows) > min_slope, slope(rows)
