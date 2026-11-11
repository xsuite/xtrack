import numpy as np
import pytest
import warnings
from scipy.spatial.transform import Rotation

import xtrack as xt


@pytest.mark.parametrize('species, charge', [('electron', -1.), ('positron', 1.)])
def test_uniform_solenoid_spin_mechanical_momentum(species, charge):
    length = 0.3
    # Keep the physical axial field fixed when reversing the charge.
    ks = charge * 1.
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
        species, p0c=5e9,
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
    line.particle_ref = xt.Particles('positron', p0c=5e9)
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


def _spin_test_ring(ks=0., p0c=5e9, qs=None, angle=0.1,
                    species="positron", anomalous_magnetic_moment=None):
    longitudinal = ({} if qs is None else dict(
        longitudinal_mode='linear_fixed_qs', qs=qs, bets=10.))
    line = xt.Line(elements=[
        xt.LineSegmentMap(qx=0.2, qy=0.3, betx=1., bety=1., length=10.,
                          **longitudinal),
        xt.Bend(length=1., angle=angle, k0=angle),
        xt.UniformSolenoid(length=0.1, ks=ks)])
    kwargs = ({} if anomalous_magnetic_moment is None else
              dict(anomalous_magnetic_moment=anomalous_magnetic_moment))
    line.particle_ref = xt.Particles(species, p0c=p0c, **kwargs)
    return line


@pytest.mark.parametrize('species, charge', [('electron', -1.), ('positron', 1.)])
@pytest.mark.parametrize('field_sign', [-1., 1.])
def test_spin_dipole_charge_analytic(species, charge, field_sign):
    # Fixed physical field magnitude: normalized curvature changes with q.
    angle = charge * field_sign * 0.02
    # Resolve the bend integration error below the analytic-test tolerance.
    line = xt.Line(elements=[xt.Bend(
        length=1., angle=angle, k0=angle, num_multipole_kicks=64)])
    line.configure_spin('auto')
    basis = np.eye(3)
    particles = xt.Particles(
        species, p0c=5e9,
        spin_x=basis[:, 0], spin_y=basis[:, 1], spin_z=basis[:, 2])
    # In the local Frenet frame the spin advances by a*gamma*angle
    # relative to the momentum. Test the full rotation on three basis spins.
    phase = (particles.anomalous_magnetic_moment[0]
             * particles.gamma0[0] * angle)
    expected = Rotation.from_rotvec([0., -phase, 0.]).apply(basis)
    line.track(particles)
    spin = np.array([particles.spin_x, particles.spin_y, particles.spin_z]).T
    np.testing.assert_allclose(spin, expected, atol=1e-11, rtol=0)
    np.testing.assert_allclose(np.linalg.norm(spin, axis=1), 1.,
                               atol=1e-12, rtol=0)
    np.testing.assert_allclose(particles.px, 0., atol=1e-12, rtol=0)
    assert np.all(particles.state == 1)


@pytest.mark.parametrize('method', ['4d', '6d'])
def test_spin_coupled_ring_charge_symmetry(method):
    results = []
    for species in ['electron', 'positron']:
        # Equal normalized strengths and opposite q mean opposite physical
        # fields, preserving q*B, orbital motion and BMT spin transport.
        line = _spin_test_ring(ks=0.01, species=species,
                               qs=0.05 if method == '6d' else None)
        line.configure_spin('auto')
        tw = line.twiss(method=method, polarization_analysis=True)
        n0 = np.array([tw.spin_x, tw.spin_y, tw.spin_z]).T
        dn = np.array([tw.spin_dn_ddelta_x, tw.spin_dn_ddelta_y,
                       tw.spin_dn_ddelta_z]).T
        times = np.array([tw.spin_t_pol_component_s,
                          tw.spin_t_pol_buildup_s,
                          tw.spin_t_depol_component_s])
        assert np.all(np.isfinite(times))
        assert np.all(times > 0)
        basis = np.eye(3)
        particles = line.build_particles(
            num_particles=3, x=1e-4, y=-2e-4, delta=1e-4,
            spin_x=basis[:, 0], spin_y=basis[:, 1], spin_z=basis[:, 2])
        line.track(particles, num_turns=32)
        spin = np.array([particles.spin_x, particles.spin_y,
                         particles.spin_z]).T
        orbit = np.array([getattr(particles, key) for key in
                          ['x', 'px', 'y', 'py', 'zeta', 'delta']])
        np.testing.assert_allclose(np.linalg.norm(spin, axis=1), 1.,
                                   atol=1e-11, rtol=0)
        assert np.max(np.abs(spin - basis)) > 0.01
        assert np.all(particles.state == 1)
        results.append((n0, dn, times, spin, orbit,
                        tw.spin_polarization_eq * n0,
                        tw.spin_polarization_inf_no_depol * n0))
    # Compare axes modulo their arbitrary global sign; physical polarization
    # vectors have no such freedom. Both charge and B reverse, so P is equal.
    sign = np.sign(np.dot(results[0][0][0], results[1][0][0]))
    for index in range(len(results[0])):
        expected = results[0][index] * (sign if index in [0, 1] else 1)
        np.testing.assert_allclose(results[1][index], expected,
                                   atol=1e-11, rtol=1e-9)


@pytest.mark.parametrize('species, charge', [('electron', -1.), ('positron', 1.)])
@pytest.mark.parametrize('angle', [-0.1, 0.1], ids=['negative_bend', 'positive_bend'])
@pytest.mark.filterwarnings('error::RuntimeWarning')
def test_polarization_charge_and_bend_direction(species, charge, angle):
    line = _spin_test_ring(angle=angle, species=species)
    tw = line.twiss4d(polarization_analysis=True)
    bend_index = list(tw.name).index(line.element_names[1])

    # The guide field has sign curvature/charge at fixed normalized geometry.
    # Sokolov-Ternov polarization is antiparallel to B for electrons and
    # parallel for positrons (Ternov, Phys. Usp. 38, 409 (1995), Sec. 3.5).
    field_y_sign = np.sign(angle / charge)
    projection = tw.spin_polarization_eq * tw.spin_y[bend_index] * field_y_sign
    limit = 8 / (5 * np.sqrt(3))
    np.testing.assert_allclose(projection, np.sign(charge) * limit,
                               rtol=1e-12, atol=0)
    np.testing.assert_allclose(tw.spin_polarization_inf_no_depol,
                               tw.spin_polarization_eq, rtol=1e-12, atol=0)
    assert np.isfinite(tw.spin_t_pol_component_s)
    assert tw.spin_t_pol_component_s > 0
    assert np.isposinf(tw.spin_t_depol_component_s)


@pytest.mark.filterwarnings('error::RuntimeWarning')
def test_polarization_4d_planar_limit():
    # The energy eigenvalue is one, so the unconstrained 3D spin solve is
    # singular. The derivative should be zero in the planar limit and vary
    # continuously as a weak solenoid is introduced.
    derivatives = []
    for ks in [0., 1e-8, 0.01]:
        line = _spin_test_ring(ks=ks, angle=0.01)
        tw = line.twiss4d(polarization_analysis=True)
        for name, value in tw._data.items():
            if name.startswith('spin_') and name not in tw._col_names:
                assert type(value) is float, name
        if ks == 0:
            assert np.isposinf(tw.spin_t_depol_component_s)
        elif ks == 0.01:
            assert np.isfinite(tw.spin_t_depol_component_s)
            assert tw.spin_t_depol_component_s > 0
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
        ks=0.01, p0c=p0c, species='proton', qs=0.05)
    tw = line.twiss(polarization_analysis=True)
    n0 = np.array([tw.spin_x, tw.spin_y, tw.spin_z]).T
    dn = tw.spin_n_matrix[:, :, 5]
    np.testing.assert_allclose(dn[-1], dn[0], atol=1e-7, rtol=0)
    np.testing.assert_allclose(np.sum(n0 * dn, axis=1), 0., atol=1e-7, rtol=0)


def test_spin_dn_ddelta_off_momentum():
    line = _spin_test_ring(
        ks=0.01, p0c=5e8, species='proton')
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
    kwargs = dict(species=species)
    if species == 'proton':
        kwargs['p0c'] = 5e8
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


@pytest.mark.parametrize('species, spin, moment', [
    ('proton', 0.5, 2.79284734463),
    ('deuteron', 1., 0.8574382335),
    ('triton', 0.5, 2.9789624650),
    ('He3', 0.5, -2.1276253498),
])
def test_nuclear_moment_solenoid_precession(species, spin, moment):
    from scipy.constants import c, e

    particle = xt.Particles(species, p0c=1e9, spin_x=1.)
    length, field = 0.3, 1.
    ks = field * c * particle.q0 / particle.p0c[0]
    line = xt.Line(elements=[xt.UniformSolenoid(
        length=length, ks=ks, edge_entry_active=False,
        edge_exit_active=False)])
    line.configure_spin('auto')
    # On axis B_parallel is unchanged by the boost. Compare the rest-frame
    # Larmor rotation -mu*B/(I*hbar) over proper time, independently of the
    # tabulated BMT anomaly. CODATA moments are in nuclear magnetons.
    proton_mass_kg = xt.PROTON_MASS_EV * e / c**2
    proper_time = length / (particle.beta0[0] * particle.gamma0[0] * c)
    phase = -moment / spin * e / (2 * proton_mass_kg) * field * proper_time
    expected = Rotation.from_rotvec([0., 0., phase]).apply([1., 0., 0.])
    line.track(particle)
    np.testing.assert_allclose(
        [particle.spin_x[0], particle.spin_y[0], particle.spin_z[0]],
        expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize('route', ['line', 'twiss'])
def test_zero_anomaly_spin_warning(route):
    if route == 'twiss':
        line = _spin_test_ring(anomalous_magnetic_moment=0.)
        with pytest.warns(UserWarning, match='anomalous_magnetic_moment=0'):
            line.twiss4d(spin=True)
    else:
        particle = xt.Particles('electron', p0c=5e9, spin_y=1.,
                                anomalous_magnetic_moment=0.)
        line = xt.Line(elements=[xt.Drift(length=1.)])
        line.configure_spin('auto')
        with pytest.warns(UserWarning, match='anomalous_magnetic_moment=0'):
            line.track(particle)


def test_spin_diagnostic_once_per_batch_and_cycle(monkeypatch):
    import gc
    from weakref import ref

    calls = 0
    original = xt.Particles._warn_if_spin_anomaly_is_zero

    def check(particle, **kwargs):
        nonlocal calls
        calls += 1
        return original(particle, **kwargs)

    monkeypatch.setattr(xt.Particles, '_warn_if_spin_anomaly_is_zero', check)
    line = xt.Line(elements=[xt.Drift(length=1.)])
    line.configure_spin('auto')
    particle = xt.Particles('electron', p0c=5e9, spin_y=1.,
                            anomalous_magnetic_moment=0.)
    with pytest.warns(UserWarning, match='anomalous_magnetic_moment=0'):
        line.track(particle)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        line.track(particle)
        line.track(particle)
    assert not caught
    assert calls == 1
    line.configure_spin('auto')
    with pytest.warns(UserWarning, match='anomalous_magnetic_moment=0'):
        line.track(particle)
    assert calls == 2
    # The one-time check must not keep a particle batch or its buffer alive.
    weak_particle = ref(particle)
    del particle
    gc.collect()
    assert weak_particle() is None


def test_spin_diagnostic_checks_alternating_batches_once(monkeypatch):
    calls = []
    original = xt.Particles._warn_if_spin_anomaly_is_zero

    def check(particle, **kwargs):
        calls.append(particle)
        return original(particle, **kwargs)

    monkeypatch.setattr(xt.Particles, '_warn_if_spin_anomaly_is_zero', check)
    line = xt.Line(elements=[xt.Drift(length=1.)])
    line.configure_spin('auto')
    batches = [xt.Particles('electron', p0c=5e9, spin_y=1.),
               xt.Particles('electron', p0c=5e9, spin_y=1.,
                            anomalous_magnetic_moment=0.)]
    line.track(batches[0])
    with pytest.warns(UserWarning, match='anomalous_magnetic_moment=0'):
        line.track(batches[1])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for particle in batches * 2:
            line.track(particle)
    assert not caught
    assert calls == batches


@pytest.mark.parametrize('route', ['co_guess', 'complete_init',
                                 'particle_on_co', 'deferred_init'])
def test_spin_diagnostic_uses_resolved_twiss_particle(route):
    line = _spin_test_ring()
    for anomaly in [0., line.particle_ref.anomalous_magnetic_moment[0]]:
        actual = xt.Particles('electron', p0c=5e9,
                              anomalous_magnetic_moment=anomaly, spin_y=1.)
        ignored = xt.Particles('electron', p0c=5e9, spin_y=1.,
                               anomalous_magnetic_moment=(0.01 if anomaly == 0 else 0.))
        line.particle_ref = ignored
        if route == 'co_guess':
            kwargs = dict(co_guess={'particle_ref': actual}, particle_ref=ignored)
        elif route == 'complete_init':
            init = xt.TwissInit(particle_on_co=actual, W_matrix=np.eye(6),
                                element_name=line.element_names[0])
            kwargs = dict(start=line.element_names[0], end=line.element_names[-1],
                          init=init, particle_on_co=ignored)
        elif route == 'particle_on_co':
            kwargs = dict(particle_on_co=actual)
        else:
            init = xt.TwissInit(betx=1., bety=1., spin_y=1.)
            kwargs = dict(start=line.element_names[0], end=line.element_names[-1],
                          init=init, particle_ref=actual)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = line.twiss4d(spin=True, **kwargs)
        messages = [item for item in caught
                    if 'anomalous_magnetic_moment=0' in str(item.message)]
        assert bool(messages) == (anomaly == 0)
        assert result.particle_on_co.anomalous_magnetic_moment[0] == anomaly


def test_bare_element_does_not_scan_spin_moments(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError('No per-element diagnostic is allowed')

    monkeypatch.setattr(xt.Particles, '_warn_if_spin_anomaly_is_zero', unexpected)
    element = xt.Drift(length=1.)
    particle = xt.Particles('electron', p0c=5e9, spin_y=1.,
                            anomalous_magnetic_moment=0.)
    element.track(particle)


def test_spin_diagnostic_does_not_read_device_arrays():
    class DeviceParticle:
        _context = object()

        def __getattr__(self, name):
            raise AssertionError(f'Device particle field read: {name}')

    xt.Particles._warn_if_spin_anomaly_is_zero(DeviceParticle())


def test_gpu_spin_setup_checks_cpu_reference(monkeypatch):
    line = xt.Line(elements=[xt.Drift(length=1.)])
    line.particle_ref = xt.Particles('electron', p0c=5e9,
                                    anomalous_magnetic_moment=0.)
    # Exercise the setup boundary without requiring a GPU installation.
    monkeypatch.setattr(xt.Line, '_context', property(lambda self: object()))
    with pytest.warns(UserWarning, match='anomalous_magnetic_moment=0'):
        line.configure_spin('auto')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        line._check_spin_moment_at_gpu_setup()
    assert not caught


def test_gpu_spin_warning_as_error_preserves_configuration(monkeypatch):
    line = xt.Line(elements=[xt.Drift(length=1.)])
    line.particle_ref = xt.Particles('electron', p0c=5e9,
                                    anomalous_magnetic_moment=0.)
    line.config.XTRACK_MULTIPOLE_NO_SYNRAD = True
    monkeypatch.setattr(xt.Line, '_context', property(lambda self: object()))
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        with pytest.raises(UserWarning, match='anomalous_magnetic_moment=0'):
            line.configure_spin('auto')
    assert line._spin_model == 'auto'
    assert not line.config.get('XTRACK_MULTIPOLE_NO_SYNRAD', False)


def test_unbuilt_cpu_line_defers_spin_diagnostic_to_tracking():
    line = xt.Line(elements=[xt.Drift(length=1.)])
    line.particle_ref = xt.Particles('electron', p0c=5e9,
                                    anomalous_magnetic_moment=0.)
    assert line._context is None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        line.configure_spin('auto')
    assert not caught
    particle = xt.Particles('electron', p0c=5e9, spin_y=1.,
                            anomalous_magnetic_moment=0.)
    with pytest.warns(UserWarning, match='anomalous_magnetic_moment=0'):
        line.track(particle)


def test_spin_diagnostic_accepts_coordinate_guess_dict():
    line = _spin_test_ring(anomalous_magnetic_moment=0.)
    with pytest.warns(UserWarning, match='anomalous_magnetic_moment=0'):
        line.twiss4d(spin=True, co_guess={'x': 0.})


@pytest.mark.parametrize('tracked', [False, True])
def test_spin_diagnostic_cache_not_serialized(tracked):
    import pickle

    line = xt.Line(elements=[xt.Drift(length=1.)])
    line.configure_spin('auto')
    if tracked:
        particle = xt.Particles('electron', p0c=5e9)
        line.track(particle)
    restored = pickle.loads(pickle.dumps(line))
    assert restored._spin_model == 'auto'
    assert '_spin_moment_checked_particles' not in restored.__dict__


@pytest.mark.parametrize('case', ['unpolarized', 'lost', 'physical', 'mixed',
                                 'Pb208', 'O16'])
def test_zero_anomaly_spin_warning_selects_live_spins(case):
    # In 'mixed', zero anomaly and nonzero spin occur on different particles.
    species = case if case in ['Pb208', 'O16'] else 'electron'
    kwargs = {
        'unpolarized': dict(anomalous_magnetic_moment=0.),
        'lost': dict(anomalous_magnetic_moment=0., spin_y=1., state=-1),
        'physical': dict(spin_y=1.),
        'mixed': dict(anomalous_magnetic_moment=[0., 0.01], spin_y=[0., 1.]),
        'Pb208': dict(spin_y=1.),
        'O16': dict(spin_y=1.),
    }[case]
    particle = xt.Particles(species, p0c=5e9, **kwargs)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        particle._warn_if_spin_anomaly_is_zero()
    assert not caught


def test_zero_anomaly_no_warning_with_spin_disabled(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError('Spin-disabled tracking must skip the diagnostic')

    monkeypatch.setattr(xt.Particles, '_warn_if_spin_anomaly_is_zero', unexpected)
    line = xt.Line(elements=[xt.Drift(length=1.)])
    line.configure_spin(None)
    particle = xt.Particles('electron', p0c=5e9, spin_y=1.,
                            anomalous_magnetic_moment=0.)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        line.track(particle)
    assert not [item for item in caught if
                'anomalous_magnetic_moment=0' in str(item.message)]


def test_zero_anomaly_diagnostic_covers_mass_catalogue():
    from xtrack.particles import pdg

    for pid, mass in pdg._get_mass_table().items():
        data = pdg.get_spin_properties_from_pdg_id(pid)
        if mass == 0 or data.anomalous_magnetic_moment is not None:
            continue
        particle = xt.Particles(pid, p0c=5e9, spin_y=1.)
        if data.spin == 0:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                particle._warn_if_spin_anomaly_is_zero()
            assert not caught
        else:
            with pytest.warns(UserWarning, match='anomalous_magnetic_moment=0'):
                particle._warn_if_spin_anomaly_is_zero()
