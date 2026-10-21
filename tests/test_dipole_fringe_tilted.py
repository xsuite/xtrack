"""Dipole fringe at tilted incidence and off momentum (xsuite/xsuite#962).

The finite-gap term of the dipole fringe generating function scales as
pz/(1+delta)^2. In the pole-face frame an on-momentum particle at incidence
angle e has pz = cos(e), so a 1/pz scaling overestimates the correction by
1/cos^2(e) (81% at 42 deg), while a pz scaling gets the momentum dependence
wrong (10% at delta = 5%).
"""

import numpy as np
import pytest
from _symplectic import assert_symplectic, fd_jacobian

import xtrack as xt

COORDS = ['x', 'px', 'y', 'py', 'zeta', 'pzeta']

B0 = 0.1  # Peak b = By/Brho [1/m]
LFR = 0.005  # Fringe extent [m]
HGAP = 0.02


def _b(z):
    u = np.clip(z / LFR, 0, 1)
    return B0 * (3 * u**2 - 2 * u**3)


def _fint():
    ss = np.linspace(0, LFR, 2001)
    bb = _b(ss)
    return np.trapezoid(bb * (B0 - bb), ss) / B0**2 / (2 * HGAP)


def _edge(fint):
    # Zero-length bend: only the entry fringe acts, in the pole-face frame
    return xt.Bend(
        length=0,
        k0=B0,
        edge_entry_model='dipole-only',
        edge_entry_fint=fint,
        edge_entry_hgap=HGAP,
        edge_exit_active=False,
    )


def _boris_edge(p):
    # Smooth fringe from b=0 to b=B0 over [0, LFR], expressed as a hard edge
    # at LFR/2: back-drift, field integration, back-bend.
    brho = p.rigidity0[0]
    a = 6 * B0 / LFR**2
    a2 = 6 * B0 / LFR**3

    def field(x, y, z):
        z = np.asarray(z)
        inside = (z >= 0) & (z <= LFR)
        bp = np.where(inside, a * z - a2 * z**2, 0)  # b'
        bpp = np.where(inside, a - 2 * a2 * z, 0)  # b''
        by = _b(z) - 0.5 * y**2 * bpp
        bz = y * bp
        return [0 * z, by * brho, bz * brho]

    xt.Line(elements=[xt.Drift(length=LFR / 2)]).track(p, backtrack=True)
    xt.BorisSpatialIntegrator(
        fieldmap_callable=field, s_start=0, s_end=LFR, n_steps=2000
    ).track(p)
    xt.Line(elements=[xt.Bend(k0=B0, length=LFR / 2, model='full')]).track(
        p, backtrack=True
    )


def _vertical_focusing(track, angle, delta):
    # d(py)/dy at y=0 for a particle crossing the face at the given angle
    dy = 1e-6
    p = xt.Particles(p0c=1e9, y=[dy, -dy], delta=delta, px=(1 + delta) * np.sin(angle))
    track(p)
    return (p.py[0] - p.py[1]) / (2 * dy)


@pytest.mark.parametrize('angle_deg', [0, 42, -42])
@pytest.mark.parametrize('delta', [-0.05, 0, 0.05])
def test_dipole_fringe_finite_gap_vs_field_integration(angle_deg, delta):
    angle = np.deg2rad(angle_deg)
    fint = _fint()

    hard = _vertical_focusing(_edge(0).track, angle, delta)
    model = _vertical_focusing(_edge(fint).track, angle, delta)
    exact = _vertical_focusing(_boris_edge, angle, delta)

    # Finite-gap correction relative to the hard edge
    corr_model = model - hard
    corr_exact = exact - hard
    assert abs(corr_exact) > 1e-6
    # 1/pz gives +81% at 42 deg, pz gives +-10% at delta=+-5%
    np.testing.assert_allclose(corr_model, corr_exact, rtol=2e-3)


def _coords(p):
    return np.array([getattr(p, cc) for cc in COORDS])


@pytest.mark.parametrize('py0', [0, 0.05])
def test_dipole_fringe_tilted_symplectic_and_reversible(py0):
    # Large angle, off momentum and py != 0 exercise every term of the
    # generating function derivatives.
    line = xt.Line(elements=[_edge(0.6)])
    line.set_particle_ref('proton', p0c=1e9)
    line.build_tracker()

    centre = dict(
        x=1e-3, px=np.sin(np.deg2rad(42)), y=2e-3, py=py0, zeta=1e-3, delta=0.03
    )
    c0 = _coords(line.build_particles(**{k: [v] for k, v in centre.items()}))[:, 0]
    assert_symplectic(fd_jacobian(line, c0))

    p = line.build_particles(**{k: [v] for k, v in centre.items()})
    p0 = p.copy()
    line.track(p)
    line.track(p, backtrack=True)
    assert np.all(p.state == 1)
    np.testing.assert_allclose(_coords(p), _coords(p0), rtol=0, atol=1e-14)

