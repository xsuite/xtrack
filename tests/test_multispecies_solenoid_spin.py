import numpy as np
import pytest

import xobjects as xo
import xtrack as xt
from xobjects.test_helpers import for_all_test_contexts


def _make_magnet(kind):
    if kind == 'legacy':
        with pytest.warns(FutureWarning, match='Solenoid.*deprecated'):
            return xt.Solenoid(length=1., ks=.5), {'ks': .5}, .5, 0., 0.
    if kind == 'variable':
        return (xt.VariableSolenoid(length=1., ks_profile=[.2, .6],
                                    x0=.003, y0=-.002),
                {'ks_profile': np.array([.2, .6])}, .2, .003, -.002)
    if kind == 'quadrupole':
        return xt.Quadrupole(length=1., k1=.5), {'k1': .5}, 0., 0., 0.
    if kind == 'bend':
        return (xt.Bend(length=1., angle=.01, k0=.01,
                        edge_entry_active=False, edge_exit_active=False),
                {'k0': .01}, 0., 0., 0.)
    active = kind == 'uniform'
    return (xt.UniformSolenoid(length=1., ks=.5, x0=.003, y0=-.002,
                               edge_entry_active=active, edge_exit_active=active),
            {'ks': .5}, 0. if active else .5, .003, -.002)


@for_all_test_contexts
@pytest.mark.parametrize('kind', [
    'uniform', 'uniform_open', 'variable', 'legacy', 'quadrupole', 'bend'])
def test_multispecies_magnet_orbit_and_spin_scaling(test_context, kind):
    magnet, strengths, entry_ks, x0, y0 = _make_magnet(kind)
    line = xt.Line(elements={'magnet': magnet})
    line.build_tracker(_context=test_context, compile=False)
    line.configure_spin(True)
    ctx2np = test_context.nparray_from_context_array
    x = np.array([.001, .02, -.01])
    y = np.array([-.002, -.01, .03])
    basis = np.eye(3)

    for charge_ratio, mass_ratio in [(2., 1.), (1., .5), (-1., 1.), (1., 2.)]:
        chi = charge_ratio / mass_ratio
        # With the same gamma and magnetic moment, chi times the field gives
        # the same orbit and BMT rotation as a particle with this chi.
        initial = xt.Particles(
            'proton', _context=test_context, p0c=1e9,
            x=x, y=y, px=2e-3, py=-1e-3, delta=[-.1, .001, .2],
            ax=-chi * entry_ks / 2 * (y - y0),
            ay=chi * entry_ks / 2 * (x - x0),
            spin_x=basis[:, 0], spin_y=basis[:, 1], spin_z=basis[:, 2])
        actual = initial.copy()
        actual.chi = chi
        actual.charge_ratio = charge_ratio
        reference = initial.copy()
        for field, value in strengths.items():
            setattr(line['magnet'], field, value)
        line.track(actual)
        for field, value in strengths.items():
            setattr(line['magnet'], field, value * chi)
        line.track(reference)
        for field in ['x', 'px', 'y', 'py', 'zeta', 'delta', 'ax', 'ay',
                      'spin_x', 'spin_y', 'spin_z']:
            xo.assert_allclose(ctx2np(getattr(actual, field)),
                               ctx2np(getattr(reference, field)),
                               rtol=0, atol=2e-12)

        # Spin backtracking is unsupported; check the reversible orbital map.
        actual.spin_x = actual.spin_y = actual.spin_z = 0.
        for field, value in strengths.items():
            setattr(line['magnet'], field, value)
        line.track(actual, backtrack=True)
        assert np.all(ctx2np(actual.state) == 1)
        for field in ['x', 'px', 'y', 'py', 'zeta', 'delta', 'ax', 'ay']:
            xo.assert_allclose(ctx2np(getattr(actual, field)),
                               ctx2np(getattr(initial, field)),
                               rtol=0, atol=2e-12)


@for_all_test_contexts
@pytest.mark.parametrize('kind', ['uniform_open', 'variable', 'legacy'])
def test_multispecies_solenoid_helix_and_axial_spin(test_context, kind):
    length, ks, delta = 1., .5, .001
    if kind == 'legacy':
        with pytest.warns(FutureWarning, match='Solenoid.*deprecated'):
            magnet = xt.Solenoid(length=length, ks=ks)
    elif kind == 'variable':
        magnet = xt.VariableSolenoid(length=length, ks_profile=[ks, ks])
    else:
        magnet = xt.UniformSolenoid(length=length, ks=ks,
                                    edge_entry_active=False, edge_exit_active=False)
    line = xt.Line(elements=[magnet])
    line.build_tracker(_context=test_context, compile=False)
    line.configure_spin(True)
    ctx2np = test_context.nparray_from_context_array
    for chi in [2., .5, -1.]:
        px, py = np.array([0., .002]), np.array([0., -.001])
        p = xt.Particles('proton', _context=test_context, p0c=1e9,
                         px=px, py=py, delta=delta, spin_x=1.,
                         charge_ratio=chi, mass_ratio=1.)
        pz = np.sqrt((1 + delta)**2 - px**2 - py**2)
        theta = chi * ks * length / pz
        expected_x = (px * np.sin(theta) + py * (1 - np.cos(theta))) / (chi * ks)
        expected_y = (py * np.sin(theta) - px * (1 - np.cos(theta))) / (chi * ks)
        expected_ax = -chi * ks / 2 * expected_y
        expected_ay = chi * ks / 2 * expected_x
        anomaly = ctx2np(p.anomalous_magnetic_moment)[0]
        phase = -(1 + anomaly) * chi * ks * length / (1 + delta)
        line.track(p)
        # The constant-field helix and on-axis BMT solution are analytic;
        # tolerances allow only accumulated floating-point roundoff.
        for field, expected in [('x', expected_x), ('y', expected_y),
                                ('ax', expected_ax), ('ay', expected_ay),
                                ('kin_px', px * np.cos(theta) + py * np.sin(theta)),
                                ('kin_py', py * np.cos(theta) - px * np.sin(theta))]:
            xo.assert_allclose(ctx2np(getattr(p, field)), expected, rtol=0, atol=2e-14)
        xo.assert_allclose([ctx2np(p.spin_x)[0], ctx2np(p.spin_y)[0], ctx2np(p.spin_z)[0]],
                           [np.cos(phase), np.sin(phase), 0.], rtol=0, atol=2e-13)


@for_all_test_contexts
def test_multispecies_solenoid_slice_boundaries(test_context):
    ks, x0, y0 = .5, .003, -.002
    line = xt.Line(elements={'sol': xt.UniformSolenoid(
        length=1., ks=ks, x0=x0, y0=y0)})
    line.build_tracker(_context=test_context, compile=False)
    chi = np.array([2., -1., .5])
    p0 = xt.Particles('proton', _context=test_context, p0c=1e9,
                      x=.01, y=-.02, px=.003, py=-.002, delta=.01,
                      charge_ratio=chi, mass_ratio=1.)
    expected = p0.copy()
    line.track(expected)
    line.discard_tracker()
    line.cut_at_s([1 / 3, 2 / 3])
    line.build_tracker(_context=test_context, compile=False)
    ctx2np = test_context.nparray_from_context_array
    p = p0.copy()
    line.track(p, ele_start='sol..entry_map', num_elements=1)
    xo.assert_allclose(ctx2np(p.ax), -chi * ks / 2 * (ctx2np(p.y) - y0),
                       rtol=0, atol=1e-15)
    xo.assert_allclose(ctx2np(p.ay), chi * ks / 2 * (ctx2np(p.x) - x0),
                       rtol=0, atol=1e-15)
    line.track(p, ele_start='sol..0')
    for field in ['x', 'px', 'y', 'py', 'zeta', 'delta', 'ax', 'ay']:
        xo.assert_allclose(ctx2np(getattr(p, field)), ctx2np(getattr(expected, field)),
                           rtol=0, atol=2e-13)
    line.track(p, backtrack=True)
    for field in ['x', 'px', 'y', 'py', 'zeta', 'delta', 'ax', 'ay']:
        xo.assert_allclose(ctx2np(getattr(p, field)), ctx2np(getattr(p0, field)),
                           rtol=0, atol=2e-13)
