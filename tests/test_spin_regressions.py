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
