import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import xtrack as xt


def test_uniform_solenoid_spin_mechanical_momentum():
    length = 0.3
    ks = 1.
    line = xt.Line(elements=[xt.UniformSolenoid(
        length=length, ks=ks, edge_entry_active=False,
        edge_exit_active=False, num_multipole_kicks=2048)])
    line.configure_spin('auto')

    # Same mechanical momentum and spin at three translated positions in a
    # uniform field. Canonical momentum changes with the vector potential.
    x = np.array([0., 0.1, -0.1])
    y = np.array([0., 0.1, -0.1])
    ax = -ks * y / 2
    ay = ks * x / 2
    p = xt.Particles(
        p0c=5e9, mass0=xt.ELECTRON_MASS_EV,
        anomalous_magnetic_moment=0.00115965218128,
        x=x, y=y, ax=ax, ay=ay, px=0.01 + ax, py=0.02 + ay,
        spin_x=1.)

    # In a frame rotating with the momentum, BMT has a constant rotation axis.
    # This gives an independent analytic solution for a uniform axial field.
    direction = np.array([0.01, 0.02, np.sqrt(1 - 0.01**2 - 0.02**2)])
    gamma = p.gamma0[0]
    g = p.anomalous_magnetic_moment[0]
    orbital_angle = -ks * length / direction[2]
    spin_rotvec = orbital_angle * g * (
        gamma * np.array([0., 0., 1.])
        + (1 - gamma) * direction[2] * direction)
    expected = Rotation.from_rotvec([0., 0., orbital_angle]).apply(
        Rotation.from_rotvec(spin_rotvec).apply([1., 0., 0.]))

    line.track(p)
    spin = np.array([p.spin_x, p.spin_y, p.spin_z]).T
    np.testing.assert_allclose(spin, np.tile(spin[0], (3, 1)), atol=1e-12, rtol=0)
    np.testing.assert_allclose(spin[0], expected, atol=2e-5, rtol=0)
    np.testing.assert_allclose(np.linalg.norm(spin, axis=1), 1., atol=1e-12, rtol=0)


@pytest.mark.parametrize('axis', ['vertical', 'tilted', 'horizontal', 'longitudinal', 'identity', 'small_angle'])
def test_spin_fixed_point_all_components(axis):
    if axis in ['longitudinal', 'identity']:
        magnet = xt.UniformSolenoid(length=0.3, ks=1. if axis == 'longitudinal' else 0.)
        expected = np.array([0., 0., 1.]) if axis == 'longitudinal' else np.array([0., 1., 0.])
    else:
        tilt = {'vertical': 0., 'tilted': 0.4, 'horizontal': np.pi / 2,
                'small_angle': 0.4}[axis]
        angle = 1e-8 if axis == 'small_angle' else 0.1
        magnet = xt.Bend(length=1., angle=angle, k0=angle, rot_s_rad=tilt)
        expected = np.array([-np.sin(tilt), np.cos(tilt), 0.])
    line = xt.Line(elements=[
        xt.LineSegmentMap(qx=0.2, qy=0.3, betx=1., bety=1., length=10.),
        magnet])
    line.particle_ref = xt.Particles(
        p0c=5e9, mass0=xt.ELECTRON_MASS_EV,
        anomalous_magnetic_moment=0.00115965218128)
    tw = line.twiss4d(spin=True)
    spin = np.array([tw.spin_x, tw.spin_y, tw.spin_z]).T
    np.testing.assert_allclose(np.linalg.norm(spin, axis=1), 1., atol=1e-12, rtol=0)
    np.testing.assert_allclose(spin[-1], spin[0], atol=1e-12, rtol=0)
    np.testing.assert_allclose(abs(np.dot(spin[0], expected)), 1., atol=1e-10, rtol=0)


@pytest.mark.parametrize('eigenvalue, reject', [(0.9995, False), (0.99, True)])
def test_spin_fixed_point_unit_eigenvalue_tolerance(monkeypatch, eigenvalue, reject):
    from xtrack.twiss import spin as spin_module

    line = xt.Line(elements=[xt.Drift(length=1.)])
    particle = xt.Particles(p0c=5e9, mass0=xt.ELECTRON_MASS_EV)
    # The actual drift preserves any spin. Perturb the supplied matrix to
    # distinguish an acceptable eigenvalue error from a missing unit eigenvalue.
    monkeypatch.setattr(spin_module, '_compute_spin_rotation_matrix',
                        lambda line, particle_on_co: np.diag([0.5, 0.8, eigenvalue]))
    if reject:
        with pytest.raises(ValueError, match='no eigenvalue sufficiently close to one'):
            spin_module._find_spin_fixed_point(line, particle)
    else:
        n0 = spin_module._find_spin_fixed_point(line, particle)
        np.testing.assert_allclose(n0, [0., 0., 1.], atol=1e-12, rtol=0)


@pytest.mark.parametrize('radiation', [None, 'mean'])
def test_configure_spin_boolean(radiation):
    line = xt.Line(elements=[xt.Bend(length=1., angle=0.1, k0=0.1)])
    line.configure_radiation(model=radiation)
    for value, expected in [(True, 'auto'), (False, None), ('auto', 'auto'),
                            (None, None)]:
        line.configure_spin(value)
        assert line._spin_model == expected
        assert line.config.get('XTRACK_MULTIPOLE_NO_SYNRAD', False) == (
            expected is None and radiation is None)
    for value in ['True', 'False', 'invalid']:
        with pytest.raises(ValueError, match='spin_model'):
            line.configure_spin(value)


def _spin_test_ring(ks=0., p0c=5e9, mass0=xt.ELECTRON_MASS_EV,
                    anomalous_magnetic_moment=0.00115965218128, qs=None, angle=0.1):
    longitudinal = ({} if qs is None else dict(
        longitudinal_mode='linear_fixed_qs', qs=qs, bets=10.))
    line = xt.Line(elements=[
        xt.LineSegmentMap(qx=0.2, qy=0.3, betx=1., bety=1., length=10.,
                          **longitudinal),
        xt.Bend(length=1., angle=angle, k0=angle),
        xt.UniformSolenoid(length=0.1, ks=ks)])
    line.particle_ref = xt.Particles(
        p0c=p0c, mass0=mass0,
        anomalous_magnetic_moment=anomalous_magnetic_moment)
    return line


@pytest.mark.parametrize('method', ['4d', '6d'])
def test_spin_rotation_matrix_reused_in_polarization(monkeypatch, method):
    from xtrack.twiss import spin as spin_module

    line = _spin_test_ring(ks=0.01, qs=0.05 if method == '6d' else None)
    compute_rotation = spin_module._compute_spin_rotation_matrix
    rotations = []

    def record_rotation(line, particle_on_co):
        rotation = compute_rotation(line, particle_on_co)
        rotations.append(rotation.copy())
        return rotation

    monkeypatch.setattr(spin_module, '_compute_spin_rotation_matrix', record_rotation)

    # Finding the spin axis already tracks three basis spins. Polarization
    # must reuse that matrix instead of tracking the same three spins again.
    tw = line.twiss(method=method, polarization_analysis=True)
    assert len(rotations) == 1

    # A supplied orbit skips the fixed-point search and needs a fresh matrix.
    # Ignore any old matrix attached to it: the lattice may have changed.
    particle_on_co = tw.particle_on_co.copy()
    particle_on_co._spin_rotation_matrix = np.full((3, 3), np.nan)
    tw_supplied = line.twiss(method=method, polarization_analysis=True,
                             particle_on_co=particle_on_co)
    assert len(rotations) == 2
    np.testing.assert_allclose(rotations[1], rotations[0], atol=1e-14, rtol=0)
    np.testing.assert_allclose(tw_supplied.spin_tune_fractional,
                               tw.spin_tune_fractional, atol=1e-14, rtol=0)
    np.testing.assert_allclose(tw_supplied.spin_n_matrix, tw.spin_n_matrix,
                               atol=1e-10, rtol=0)

    # Reuse is confined to one Twiss call, not cached on the line.
    line.elements[-1].ks *= 2
    line.twiss(method=method, polarization_analysis=True)
    assert len(rotations) == 3
    assert np.max(np.abs(rotations[2] - rotations[0])) > 1e-5


def test_polarization_4d_planar_limit():
    # The energy eigenvalue is one, so the unconstrained 3D spin solve is
    # singular. The derivative should be zero in the planar limit and vary
    # continuously as a weak solenoid is introduced.
    derivatives = []
    for ks in [0., 1e-8, 0.01]:
        line = _spin_test_ring(ks=ks, angle=0.01)
        tw = line.twiss4d(polarization_analysis=True)
        n0 = np.array([tw.spin_x, tw.spin_y, tw.spin_z]).T
        dn = tw.spin_n_matrix[:, :, 5]
        assert np.all(np.isfinite(dn))
        np.testing.assert_allclose(dn[-1], dn[0], atol=1e-7, rtol=0)
        np.testing.assert_allclose(np.sum(n0 * dn, axis=1), 0., atol=1e-7, rtol=0)
        derivatives.append(dn[0])
    np.testing.assert_allclose(derivatives[0], 0., atol=1e-10, rtol=0)
    np.testing.assert_allclose(derivatives[1], derivatives[0], atol=1e-7, rtol=0)
    assert np.linalg.norm(derivatives[2]) > 1e-5


@pytest.mark.parametrize('p0c', [5e8, 2e9, 1e12])
def test_spin_polarization_energy_coordinates(p0c):
    line = _spin_test_ring(
        ks=0.01, p0c=p0c, mass0=xt.PROTON_MASS_EV,
        anomalous_magnetic_moment=1.79284734463, qs=0.05)
    tw = line.twiss(polarization_analysis=True)
    n0 = np.array([tw.spin_x, tw.spin_y, tw.spin_z]).T
    dn = tw.spin_n_matrix[:, :, 5]
    np.testing.assert_allclose(dn[-1], dn[0], atol=1e-7, rtol=0)
    np.testing.assert_allclose(np.sum(n0 * dn, axis=1), 0., atol=1e-7, rtol=0)


def test_spin_dn_ddelta_off_momentum():
    line = _spin_test_ring(
        ks=0.01, p0c=5e8, mass0=xt.PROTON_MASS_EV,
        anomalous_magnetic_moment=1.79284734463)
    delta0 = 0.1
    step = 1e-5
    tw = line.twiss4d(delta0=delta0, polarization_analysis=True)
    plus = line.twiss4d(delta0=delta0 + step, spin=True)
    minus = line.twiss4d(delta0=delta0 - step, spin=True)
    d_orbit = np.array([(plus[k][0] - minus[k][0]) / (2 * step)
                        for k in ['x', 'px', 'y', 'py']])
    d_spin = np.array([(plus[k][0] - minus[k][0]) / (2 * step)
                       for k in ['spin_x', 'spin_y', 'spin_z']])
    # Varying the closed-orbit energy also changes the transverse orbit.
    # Include that contribution when comparing to independent spin solutions.
    predicted = tw.spin_n_matrix[0, :, :4] @ d_orbit + np.array([
        tw.spin_dn_ddelta_x[0], tw.spin_dn_ddelta_y[0], tw.spin_dn_ddelta_z[0]])
    np.testing.assert_allclose(predicted, d_spin, atol=2e-7, rtol=2e-5)


def _spin_field_tracking_errors(line, tw, offsets, nn):
    coordinates = ['x', 'px', 'y', 'py', 'zeta', 'pzeta']
    co = np.array([tw[key][0] for key in coordinates])
    n0 = np.array([tw.spin_x[0], tw.spin_y[0], tw.spin_z[0]])
    spin = n0 + offsets @ nn.T
    spin /= np.linalg.norm(spin, axis=1)[:, None]
    particles = line.build_particles(
        particle_ref=tw.particle_on_co, mode='set',
        **{key: co[i] + offsets[:, i] for i, key in enumerate(coordinates)},
        spin_x=spin[:, 0], spin_y=spin[:, 1], spin_z=spin[:, 2])

    # Track the same dynamics as Twiss. Particle zero is on the closed orbit
    # and supplies the common radiation kicks when using kick_as_co.
    with xt.line._preserve_config(line), xt.line._preserve_track_flags(line):
        line.config.XTRACK_MULTIPOLE_NO_SYNRAD = False
        line.tracker.track_flags.XS_FLAG_KILL_CAVITY_KICK = tw.method == '4d'
        line.tracker.track_flags.XS_FLAG_SR_KICK_SAME_AS_FIRST = (
            tw.radiation_method == 'kick_as_co')
        line.track(particles)
    np.testing.assert_array_equal(particles.state, 1)
    tracked_spin = np.array([
        particles.spin_x, particles.spin_y, particles.spin_z]).T
    np.testing.assert_allclose(
        np.linalg.norm(tracked_spin, axis=1), 1., atol=1e-12, rtol=0)

    # Use the actual final orbit, not its linear approximation R @ offsets.
    # After a full turn, predict spin with the INITIAL n0 and NN at the same
    # azimuth. Using the propagated end-point NN would weaken this invariance
    # check by reusing the eigenvector propagation that we want to validate.
    final_offsets = np.array([
        getattr(particles, key) for key in coordinates]).T - co
    predicted_spin = n0 + final_offsets @ nn.T
    predicted_spin /= np.linalg.norm(predicted_spin, axis=1)[:, None]
    return np.linalg.norm(tracked_spin - predicted_spin, axis=1)


def _check_linear_spin_field_tracking(line, tw, scales, max_control_error):
    # For a correct linear invariant spin field n_lin(u) = normalize(n0 + NN u),
    # tracking n_lin(u) must agree with n_lin(u_final) through first order.
    # The omitted nonlinear terms give an O(epsilon**2) error: halving every
    # orbital offset should reduce this error by four. A wrong NN instead
    # generally leaves an O(epsilon) error, which only decreases by two.
    # Probe each coordinate separately, both signs, and a mixed displacement.
    directions = np.vstack([np.eye(6), [1., -0.7, 0.4, 0.9, -0.5, 0.8]])
    offsets = scales * np.vstack([np.zeros((2, 6)), directions, -directions])
    amplitudes = [1., 0.5, 0.25]
    nn = tw.spin_n_matrix[0]
    errors = np.array([
        _spin_field_tracking_errors(line, tw, amplitude * offsets, nn)
        for amplitude in amplitudes])

    # Two unperturbed controls distinguish closure error (particle zero) from
    # the numerical floor affecting the other particles. In kick_as_co, the
    # first particle's energy update and applying its copied kick have slightly
    # different roundoff, which can accumulate over a long electron ring.
    np.testing.assert_allclose(errors[:, 0], 0., atol=1e-11, rtol=0)
    np.testing.assert_allclose(errors[:, 1], 0., atol=max_control_error, rtol=0)
    floor = max(2e-12, 2 * np.max(errors[:, 1]))
    assert np.all(errors[1:, 2:] <= 0.30 * errors[:-1, 2:] + floor)
    peak_errors = np.max(errors[:, 2:], axis=1)
    assert peak_errors[-1] > 3 * floor  # Resolve the nonlinear error above noise.
    np.testing.assert_allclose(
        peak_errors[1:] / peak_errors[:-1], 0.25, atol=0.04, rtol=0)

    # Negative control: an intentionally incorrect field must exhibit the
    # linear error that the quadratic-convergence assertion above rejects.
    wrong_errors = np.array([
        _spin_field_tracking_errors(line, tw, amplitude * offsets, 0.9 * nn)
        for amplitude in amplitudes])
    wrong_peaks = np.max(wrong_errors[:, 2:], axis=1)
    np.testing.assert_allclose(
        wrong_peaks[1:] / wrong_peaks[:-1], 0.5, atol=0.05, rtol=0)
    assert np.any(wrong_errors[1:, 2:] > 0.30 * wrong_errors[:-1, 2:] + floor)


@pytest.mark.parametrize('method', ['4d', '6d'])
@pytest.mark.parametrize('species', ['electron', 'proton'])
def test_linear_spin_field_tracking(method, species):
    kwargs = ({} if species == 'electron' else dict(
        p0c=5e8, mass0=xt.PROTON_MASS_EV,
        anomalous_magnetic_moment=1.79284734463))
    line = _spin_test_ring(ks=0.01, qs=0.05 if method == '6d' else None, **kwargs)
    tw = line.twiss(method=method, polarization_analysis=True)
    # x, px, y, py, zeta, pzeta: small finite amplitudes, deliberately larger
    # than the finite-difference steps used internally to calculate NN.
    scales = np.array([1e-3, 2e-4, 1.2e-3, 3e-4, 2e-3, 1e-3])
    _check_linear_spin_field_tracking(line, tw, scales, max_control_error=2e-12)


@pytest.mark.filterwarnings('ignore::FutureWarning')
def test_linear_spin_field_tracking_with_radiation():
    from pathlib import Path

    line = xt.load(Path(__file__).parent
                   / '../test_data/spin_refs_bmad/lep_lattice_to_bmad.json')
    for ip in [2, 4, 6, 8]:
        line[f'on_sol.{ip}'] = 1
        for knob in ['on_spin_bump', 'on_coupl_sol', 'on_coupl_sol_bump']:
            line[f'{knob}.{ip}'] = 0
    line.configure_radiation(model='mean')
    line.compensate_radiation_energy_loss()
    tw = line.twiss(polarization_analysis=True)
    assert tw.radiation_method == 'kick_as_co'
    # Keep the quadratic error above the roundoff floor of the long ring.
    scales = 4 * np.array([1e-5, 1e-7, 1e-5, 1e-7, 1e-4, 1e-5])
    _check_linear_spin_field_tracking(line, tw, scales, max_control_error=5e-8)
