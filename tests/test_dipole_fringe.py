# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

import numpy as np
import pytest
from scipy.integrate import solve_ivp

import xobjects as xo
import xtrack as xt
from xobjects.test_helpers import for_all_test_contexts, skip_if_forbid_compile


@pytest.fixture
def fringe_kernel_compilation(from_ptc):
    # Only the optional debugging configuration lacks prebuilt kernels.
    # Default-map tests use the normal kernel-selection policy unchanged.
    if from_ptc:
        skip_if_forbid_compile()
        with xo.settings.override(allow_kernel_compilation=True):
            yield
    else:
        yield


def _integrated_fringe_correction(k0, length, angle, delta):
    """Linear vertical optics through a cubic field, independent of the map.

    At first order in y and py, Maxwell's equations give B_s/Brho=y*b'(s).
    Unit y is a transfer-matrix normalization, not a physical 1 m offset.
    Removing the upstream drift (initial py=0) and downstream uniform bend
    does not change the measured py/y. Subtract the hard-edge contribution.
    """
    momentum = 1 + delta
    px = momentum * np.sin(angle)

    def rhs(s, state):
        px_local, y, py = state
        u = s / length
        b = k0 * (3*u**2 - 2*u**3)
        db = k0 / length * (6*u - 6*u**2)
        pz = np.sqrt(momentum**2 - px_local**2)
        return [-b, py / pz, -px_local / pz * y * db]

    solution = solve_ivp(rhs, [0, length], [px, 1., 0.],
                         rtol=1e-11, atol=1e-13)
    assert solution.success
    return solution.y[2, -1] + k0 * np.tan(angle)


@pytest.mark.parametrize('from_ptc', [False, True])
@for_all_test_contexts
def test_dipole_fringe_against_field_integration(
        test_context, from_ptc, fringe_kernel_compilation):
    """Test the finite-gap correction, rather than the much larger total kick."""
    angles, deltas = np.meshgrid(np.deg2rad([-42., 0., 42.]), [-.05, 0., .05])
    angles, deltas = angles.ravel(), deltas.ravel()
    line = xt.Line(elements=[xt.DipoleEdge(k=.1, model='full', hgap=.02)])
    line.config.XTRACK_FRINGE_FROM_PTC = from_ptc
    line.build_tracker(_context=test_context)
    y0 = 1e-7

    for k0 in [.1, -.1]:
        line[0].k = k0
        errors = []
        for length in [.05, .005]:
            # integral f*(1-f) ds = 9*length/70; full gap = 2*hgap.
            line[0].fint = 9*length / (140*line[0].hgap)
            particles = xt.Particles(
                _context=test_context, p0c=1e9, y=y0,
                px=(1+deltas)*np.sin(angles), delta=deltas)
            line.track(particles)
            correction = test_context.nparray_from_context_array(particles.py) / y0
            correction += k0 * np.tan(angles)
            integrated = np.array([
                _integrated_fringe_correction(k0, length, angle, delta)
                for angle, delta in zip(angles, deltas)])
            error = np.abs(correction / integrated - 1)
            errors.append(error)
            assert np.all(error < (2e-3 if length == .05 else 2e-4))
        # The finite-width remainder decreases towards the thin-fringe limit.
        tilted = angles != 0
        assert np.all(errors[1][tilted] < .11 * errors[0][tilted])


def _fringe_psi(px, py, ptau, beta0, k0, hgap, fint):
    """Generating coefficient; complex inputs allow independent derivatives."""
    pz = np.sqrt(1 + 2*ptau/beta0 + ptau**2 - px**2 - py**2)
    xp, yp = px/pz, py/pz
    u = xp / (1 + yp**2)
    a = 1 + xp**2 * (2 + yp**2)
    return k0 * np.tan(np.arctan(u) - 2*k0*hgap*fint*a/(pz*(1+u**2)))


@pytest.mark.parametrize('from_ptc', [False, True])
@for_all_test_contexts
def test_dipole_fringe_generating_function(
        test_context, from_ptc, fringe_kernel_compilation):
    """Check all derivatives, including nonzero py and the longitudinal map."""
    beta0, hgap = .5, .025
    line = xt.Line(elements=[xt.DipoleEdge(k=.12, model='full', hgap=hgap)])
    line.particle_ref = xt.Particles(beta0=beta0)
    line.config.XTRACK_FRINGE_FROM_PTC = from_ptc
    line.build_tracker(_context=test_context)

    for k0, fint in [(.12, .9), (-.12, .9), (.12, 0.)]:
        line[0].k, line[0].fint = k0, fint
        initial = xt.Particles(
            beta0=beta0, x=[.01, -.02, .03], y=[.002, -.003, .001],
            px=[.6, -.3, .15], py=[.2, -.3, .1], delta=[-.05, .1, 0.],
            zeta=[.01, .02, -.03])
        momenta = [initial.px, initial.py, initial.ptau]
        psi = _fringe_psi(*momenta, beta0, k0, hgap, fint)
        derivatives = []
        for index in range(3):
            shifted = [np.array(p, dtype=complex) for p in momenta]
            shifted[index] += 1e-30j
            derivatives.append(
                _fringe_psi(*shifted, beta0, k0, hgap, fint).imag / 1e-30)
        psi_px, psi_py, psi_ptau = derivatives
        yf = 2*initial.y / (1 + np.sqrt(1 - 2*psi_py*initial.y))
        momentum = 1 + initial.delta
        c3 = k0**2 / (72*hgap*fint*momentum) if fint else np.zeros(3)
        expected = dict(
            x=initial.x + .5*psi_px*yf**2,
            px=initial.px,
            y=yf,
            py=initial.py - psi*yf - 4*c3*yf**3,
            zeta=initial.zeta + beta0*(.5*psi_ptau*yf**2
                  - c3*(1/beta0 + initial.ptau)/momentum**2*yf**4),
            delta=initial.delta)
        particles = initial.copy(_context=test_context)
        line.track(particles)
        for name, value in expected.items():
            actual = test_context.nparray_from_context_array(getattr(particles, name))
            xo.assert_allclose(actual, value, rtol=0, atol=2e-14)

        # Full canonical symplecticity, rather than only det(R)=1.
        reference = initial.copy(_context=test_context)
        reference = reference.filter(reference.particle_id == 0)
        matrix = line.get_R_matrix(particle_on_co=reference)['R_matrix']
        symplectic = np.kron(np.eye(3), [[0., 1.], [-1., 0.]])
        xo.assert_allclose(matrix.T @ symplectic @ matrix, symplectic,
                           rtol=0, atol=2e-8)
        if not from_ptc:
            line.track(particles, backtrack=True)
            for name in expected:
                actual = test_context.nparray_from_context_array(getattr(particles, name))
                xo.assert_allclose(actual, getattr(initial, name), rtol=0, atol=2e-14)


@pytest.mark.parametrize('from_ptc', [False, True])
@for_all_test_contexts
def test_dipole_fringe_tilted_ring(
        test_context, from_ptc, fringe_kernel_compilation):
    """Four 90-degree bends with 42-degree pole faces: reported regression."""
    elements = []
    for k1 in [0.8581336301, 0.8582382342] * 2:
        elements.extend([
            xt.Drift(length=.1),
            xt.Bend(length=2.2, angle=1.570796327,
                    edge_entry_angle=np.deg2rad(42), edge_exit_angle=np.deg2rad(42),
                    edge_entry_hgap=.025, edge_exit_hgap=.025,
                    edge_entry_fint=.9, edge_exit_fint=.9),
            xt.Drift(length=1.), xt.Quadrupole(length=.2, k1=k1),
            xt.Drift(length=.9)])
    line = xt.Line(elements=elements)
    line.particle_ref = xt.Particles(beta0=.5863443683017825)
    line.configure_bend_model(core='full', edge='full')
    line.configure_drift_model('exact')
    line.config.XTRACK_FRINGE_FROM_PTC = from_ptc
    line.build_tracker(_context=test_context)
    tw = line.twiss(method='4d')
    xo.assert_allclose([tw.qx, tw.qy], [.683381030912, 1.423022189857],
                       rtol=0, atol=2e-9)
    # The on-momentum tune is restored, but correcting momentum dependence
    # intentionally does not restore the old vertical chromaticity (~0.5796).
    xo.assert_allclose([tw.dqx, tw.dqy], [.6670713, .8181534234],
                       rtol=0, atol=5e-6)


@for_all_test_contexts
def test_dipole_fringe_on_momentum_against_ptc(test_context, sandbox_cwd):
    """Compare the reported ring with external, unmodified MAD-X/PTC.

    On the mid-plane at delta=0 the corrected fringe factor equals the
    original PTC factor. This agreement is expected for transverse linear
    optics, not for chromaticities or the general nonlinear map.
    """
    from cpymad.madx import Madx

    beta0 = .5863443683017825
    with Madx(stdout=False) as mad:
        mad.input(f'''
            beam, particle=proton, beta={beta0};
            d1: drift, l=.1;
            d2: drift, l=1.;
            d3: drift, l=.9;
            b: sbend, l=2.2, angle=1.570796327,
                e1=42*pi/180, e2=42*pi/180,
                hgap=.025, fint=.9, fintx=.9;
            q1: quadrupole, l=.2, k1=.8581336301;
            q2: quadrupole, l=.2, k1=.8582382342;
            ring: line=(d1,b,d2,q1,d3, d1,b,d2,q2,d3,
                        d1,b,d2,q1,d3, d1,b,d2,q2,d3);
            use, period=ring;
            ptc_create_universe;
            ptc_create_layout, time=false, model=1, exact=true,
                               method=6, nst=100;
            ptc_twiss, closed_orbit, icase=56, no=1, deltap=0;
        ''')
        ptc_tunes = np.array([mad.table.ptc_twiss.mu1[-1],
                              mad.table.ptc_twiss.mu2[-1]])
        line = xt.Line.from_madx_sequence(mad.sequence.ring, allow_thick=True)
        mad.input('ptc_end;')

    line.particle_ref = xt.Particles(beta0=beta0)
    line.configure_bend_model(core='full', edge='full')
    line.configure_drift_model('exact')
    # Test the default implementation against external PTC, not the optional
    # PTC-derived Xtrack branch (which already contains our correction).
    line.config.XTRACK_FRINGE_FROM_PTC = False
    line.build_tracker(_context=test_context)
    tw = line.twiss(method='4d', delta0=0)
    xo.assert_allclose([tw.qx, tw.qy], ptc_tunes, rtol=0, atol=2e-9)


@pytest.mark.parametrize('px, py, delta, y', [
    (.3, .2, .05, .002),
    (-.3, -.3, .1, -.003),
    (.15, .1, -.05, .001),
])
def test_dipole_fringe_tpsa(px, py, delta, y):
    """Check TPSA derivatives and symplecticity with the cubic fringe term."""
    from xtrack.tpsa import ParticlesTpsa

    line = xt.Line(elements=[xt.Bend(
        length=0, k0=.12, edge_entry_model='full',
        edge_entry_angle=.3, edge_entry_hgap=.025, edge_entry_fint=.9,
        edge_exit_active=False)])
    line.build_tracker()
    names = ('x', 'px', 'y', 'py', 'zeta', 'delta')
    seed = np.array([.01, px, y, py, .01, delta])
    tpsa = ParticlesTpsa(order=2, p0c=1e9, **dict(zip(names, seed)))
    line.track(tpsa)

    # Central differences in (x, px, y, py, zeta, delta), the TPSA coordinates.
    step = 1e-6
    coords = np.column_stack([seed, seed[:, None] + step*np.eye(6),
                             seed[:, None] - step*np.eye(6)])
    particles = xt.Particles(p0c=1e9, **dict(zip(names, coords)))
    line.track(particles)
    tracked = np.array([getattr(particles, name) for name in names])
    jacobian = (tracked[:, 1:7] - tracked[:, 7:13]) / (2*step)
    xo.assert_allclose(tpsa.const_part, tracked[:, 0], rtol=0, atol=2e-14)
    xo.assert_allclose(tpsa.jacobian(), jacobian, rtol=0, atol=2e-8)

    # TPSA uses delta, whereas the canonical momentum conjugate to zeta is
    # p_zeta = ptau/beta0. Its derivative is dp_zeta/ddelta = beta/beta0
    # = rvv. Energy is conserved by this map, so the input/output factors
    # coincide. Transform both the output row and the input column.
    canonical_jacobian = tpsa.jacobian().copy()
    rvv = particles.rvv[0]
    canonical_jacobian[5, :] *= rvv
    canonical_jacobian[:, 5] /= rvv
    symplectic = np.kron(np.eye(3), [[0., 1.], [-1., 0.]])
    xo.assert_allclose(
        canonical_jacobian.T @ symplectic @ canonical_jacobian,
        symplectic, rtol=0, atol=1e-13)
