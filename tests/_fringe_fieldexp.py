"""Hard-edge fringe maps against a field of smooth ramps.

Reference: a magnet whose strength rises from 0 over a ramp of length ell (C2
smoothstep), integrated through its field expansion. Model: the hard-edge
magnet, faces at the ramp midpoints, with the fringe map under test.

Maps are TPSA maps around the zero orbit (exact Taylor coefficients, no fit).
Compared: the coefficients of one degree and one order in the strength, the
term the fringe map models. Accepted when the fringe removes the error and the
residual falls with ell at the expected rate.
"""
from math import factorial

import numpy as np
from numpy.polynomial import Polynomial

import xtrack as xt
import xtrack.tpsa as xtpsa

SMOOTHSTEP = Polynomial([0, 0, 0, 10, -15, 6])
COORDS = ('x', 'px', 'y', 'py')
P0C = 1e9


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


def make_line(elements):
    line = xt.Line(elements=elements)
    line.set_particle_ref('proton', p0c=P0C)
    line.build_tracker()
    return line


def tpsa_map(line, degree):
    """Coefficients of ``degree`` of the transverse map at delta = 0, as
    {(coordinate, exponents of x, px, y, py): value}."""
    m = xtpsa.ParticlesTpsa(order=degree, p0c=P0C, mass0=xt.PROTON_MASS_EV,
                            x=0., px=0., y=0., py=0., zeta=0., delta=0.)
    line.track(m)
    return {(c, mono[:4]): val
            for c in COORDS
            for mono, val in getattr(m, c).monomial_coeffs(tol=0.).items()
            if sum(mono[:4]) == degree and not any(mono[4:])}


def strength_part(build, power, degree, h=1e-2):
    """Coefficients of ``degree`` at order ``power`` (1 or 2) in the strength;
    4th-order central differences of ``build(k)``, error O(h^4)."""
    weights = {1: (1, -8, 0, 8, -1), 2: (-1 / 2, 8, -15, 8, -1 / 2)}[power]
    out = {}
    for k, weight in zip((-2 * h, -h, 0., h, 2 * h), weights):
        if weight == 0:
            continue
        for kk, vv in tpsa_map(build(k), degree).items():
            out[kk] = out.get(kk, 0.) + weight * vv / (12 * h**power)
    return out


def rel_diff(coeffs, reference):
    keys = set(coeffs) | set(reference)
    diff = max(abs(coeffs.get(kk, 0.) - reference.get(kk, 0.)) for kk in keys)
    return diff / max(abs(vv) for vv in reference.values())


def slope(ells, errs):
    return np.polyfit(np.log(ells), np.log(errs), 1)[0]
