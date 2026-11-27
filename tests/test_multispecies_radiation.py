import numpy as np
import pytest
from scipy.constants import alpha, c, e, epsilon_0, hbar

import xobjects as xo
import xtrack as xt
from xobjects.test_helpers import for_all_test_contexts


@for_all_test_contexts
@pytest.mark.parametrize('thick', [False, True])
def test_multispecies_radiation_mean_scaling(test_context, thick):
    if thick:
        element = xt.Bend(length=1., angle=.01, k0_from_h=True,
                          edge_entry_active=False, edge_exit_active=False)
    else:
        element = xt.Multipole(length=1., knl=[.01], hxl=.01)
    line = xt.Line(elements=[element])
    line.build_tracker(_context=test_context)
    line.configure_radiation(model='mean')

    charge_ratio = np.array([1., 2., 1., -2., .5])
    mass_ratio = np.array([1., 2., 2., 2., 1.])
    for delta in [0., -.2, .3]:
        particles = xt.Particles(
            _context=test_context, mass0=xt.ELECTRON_MASS_EV, q0=-1,
            p0c=10e9, delta=delta,
            charge_ratio=charge_ratio, mass_ratio=mass_ratio)
        gamma0 = test_context.nparray_from_context_array(particles.gamma0)[0]
        radius = e / (4 * np.pi * epsilon_0 * xt.ELECTRON_MASS_EV)
        # At fixed field and gamma, dE/E scales as q^4/m^3. Delta is
        # normalized by mass_ratio*p0c; the off-momentum factor is (1+delta)^2.
        expected = (-2 / 3 * radius * gamma0**3 * .01**2
                    * charge_ratio**4 / mass_ratio**3 * (1 + delta)**2)
        line.track(particles)
        actual = test_context.nparray_from_context_array(particles.delta) - delta
        # The analytic estimate uses the design length, whereas tracking uses
        # the actual curved path, differing by O(angle^2) for off-rigidity particles.
        xo.assert_allclose(actual, expected, rtol=5e-4, atol=2e-16)


@for_all_test_contexts
@pytest.mark.parametrize('model', ['mean', 'quantum', 'quantum-kick'])
def test_multispecies_radiation_reference_independence(test_context, model):
    line = xt.Line(elements={'bend': xt.Bend(
        length=1., angle=.01, k0=.01,
        edge_entry_active=False, edge_exit_active=False)})
    line.build_tracker(_context=test_context)
    line.configure_radiation(model=model)
    n_part = 128
    seeds = np.arange(n_part, dtype=np.uint32) + 1234
    delta = np.linspace(-.2, .3, n_part)
    ctx2np = test_context.nparray_from_context_array

    for charge_ratio, mass_ratio in [(2., 2.), (1., 2.), (-2., 2.)]:
        mixed = xt.Particles(
            _context=test_context, mass0=xt.ELECTRON_MASS_EV, q0=-1,
            p0c=10e9, delta=delta, px=1e-4, py=-2e-4,
            charge_ratio=charge_ratio, mass_ratio=mass_ratio)
        own = xt.Particles(
            _context=test_context, mass0=xt.ELECTRON_MASS_EV * mass_ratio,
            q0=-charge_ratio, p0c=10e9 * mass_ratio,
            delta=delta, px=1e-4, py=-2e-4)
        mixed._init_random_number_generator(seeds=seeds)
        own._init_random_number_generator(seeds=seeds)
        xo.assert_allclose(ctx2np(mixed.energy), ctx2np(own.energy), rtol=1e-14)

        line['bend'].k0 = .01
        line.track(mixed)
        # Preserve the physical field when changing the reference rigidity.
        line['bend'].k0 = .01 * charge_ratio / mass_ratio
        line.track(own)
        for coordinate in ['x', 'px', 'y', 'py', 'zeta', 'delta']:
            xo.assert_allclose(ctx2np(getattr(mixed, coordinate)),
                               ctx2np(getattr(own, coordinate)),
                               rtol=1e-10, atol=2e-13)
        xo.assert_allclose(ctx2np(mixed.energy), ctx2np(own.energy), rtol=1e-13)


@for_all_test_contexts
@pytest.mark.parametrize('model', ['quantum', 'quantum-kick'])
@pytest.mark.parametrize('q0', [-1., 2.])
def test_multispecies_radiation_quantum_mean(test_context, model, q0):
    line = xt.Line(elements=[xt.Bend(
        length=1., angle=.01, k0_from_h=True,
        edge_entry_active=False, edge_exit_active=False)])
    line.build_tracker(_context=test_context)
    charge_ratio = np.array([1., 2., 1., -2.])
    mass_ratio = np.array([1., 2., 2., 2.])
    n_part = 40000
    ctx2np = test_context.nparray_from_context_array
    initial = xt.Particles(
        _context=test_context, mass0=xt.ELECTRON_MASS_EV, q0=q0,
        p0c=10e9, delta=0., charge_ratio=charge_ratio, mass_ratio=mass_ratio)
    line.configure_radiation(model='mean')
    line.track(initial)
    expected = ctx2np(initial.delta)

    particles = xt.Particles(
        _context=test_context, mass0=xt.ELECTRON_MASS_EV, q0=q0,
        p0c=10e9, delta=0.,
        charge_ratio=np.repeat(charge_ratio, n_part),
        mass_ratio=np.repeat(mass_ratio, n_part))
    particles._init_random_number_generator(
        seeds=np.arange(len(charge_ratio) * n_part, dtype=np.uint32) + 1)
    line.configure_radiation(model=model)
    line.track(particles)
    losses = ctx2np(particles.delta).reshape(len(charge_ratio), n_part)
    # Six standard errors cover sampling noise. The 0.5% allowance also
    # covers the inverse-CDF approximation used by quantum-kick.
    tolerance = (6 * losses.std(axis=1) / np.sqrt(n_part)
                 + .005 * np.abs(expected))
    assert np.all(np.abs(losses.mean(axis=1) - expected) < tolerance)


@for_all_test_contexts
def test_multispecies_radiation_photon_energy_record(test_context):
    line = xt.Line(elements=[xt.Multipole(length=1., knl=[.01], hxl=.01)])
    line.build_tracker(_context=test_context)
    line.configure_radiation(model='quantum')
    record = line.start_internal_logging_for_elements_of_type(xt.Multipole, capacity=50000)
    n_part = 2000
    particles = xt.Particles(
        _context=test_context, mass0=xt.ELECTRON_MASS_EV, q0=-1,
        p0c=10e9, delta=0., charge_ratio=2., mass_ratio=2., x=np.zeros(n_part))
    particles._init_random_number_generator(
        seeds=np.arange(n_part, dtype=np.uint32) + 9876)
    ctx2np = test_context.nparray_from_context_array
    initial_energy = ctx2np(particles.energy).copy()
    gamma0 = ctx2np(particles.gamma0)[0]
    line.track(particles)
    line.stop_internal_logging_for_elements_of_type(xt.Multipole)
    record.move(_context=xo.ContextCpu())
    count = record._index.num_recorded
    assert 0 < count < 50000
    # Here chi=1, so the curvature is .01/m, but the charge is 2e.
    expected_count = n_part * 2.5 / np.sqrt(3) * alpha * 2**2 * gamma0 * .01
    assert abs(count - expected_count) < 6 * np.sqrt(expected_count)
    critical_energy = 1.5 * hbar * c / e * gamma0**3 * .01
    expected_photon_energy = 8 * np.sqrt(3) / 45 * critical_energy
    photons = record.photon_energy[:count]
    assert abs(np.mean(photons) - expected_photon_energy) < (
        6 * np.std(photons) / np.sqrt(count) + 1e-5 * expected_photon_energy)
    # Recorded photon energies must sum to the physical loss, including the
    # mass_ratio factor in the conversion from normalized energy coordinates.
    xo.assert_allclose(np.sum(initial_energy - ctx2np(particles.energy)),
                       np.sum(record.photon_energy[:count]), rtol=1e-8, atol=0)
