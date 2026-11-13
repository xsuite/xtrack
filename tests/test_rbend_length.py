import numpy as np
import pytest

import xtrack as xt


@pytest.mark.parametrize('model', ['curved-body', 'straight-body'])
@pytest.mark.parametrize('angle', [0., -1e-12, 1e-12, -1e-10, 1e-10, -1e-9, 1e-9])
@pytest.mark.parametrize('angle_diff', [0., -np.deg2rad(22.), np.deg2rad(22.)])
def test_rbend_length_near_zero_angle(model, angle, angle_diff):
    length_straight = 0.4713
    bend = xt.RBend(length_straight=length_straight, angle=angle,
                    rbend_angle_diff=angle_diff, rbend_model=model)
    # At these angles, the O(angle**2) arc-length correction is below
    # double precision. The inclined straight trajectory is the reference.
    expected_length = length_straight / np.cos(angle_diff / 2)
    np.testing.assert_allclose(bend.length, expected_length, rtol=2e-15, atol=0)
    np.testing.assert_allclose(bend.h, angle / expected_length, rtol=2e-15, atol=0)
    assert bend._k0 == bend.h

    line = xt.Line(elements=[bend, xt.Drift(length=1.)])
    expected_s = [0., expected_length, expected_length + 1.]
    np.testing.assert_allclose(line.get_table().s, expected_s, rtol=0, atol=1e-15)
    survey = line.survey()
    np.testing.assert_allclose(survey.Z, expected_s, rtol=0, atol=1e-15)


@pytest.mark.parametrize('angle', [-np.pi / 3, -0.1, 0.1, np.pi / 3])
@pytest.mark.parametrize('angle_diff', [-0.3, 0., 0.3])
def test_rbend_length_finite_angle(angle, angle_diff):
    bend = xt.RBend(length_straight=0.7, angle=angle, rbend_angle_diff=angle_diff)
    # Check the entrance/exit projection relation independently of sinc.
    theta_in = (angle - angle_diff) / 2
    theta_out = (angle + angle_diff) / 2
    expected_h = (np.sin(theta_in) + np.sin(theta_out)) / 0.7
    np.testing.assert_allclose(bend.h, expected_h, rtol=2e-15, atol=0)
    np.testing.assert_allclose(bend.length, angle / expected_h, rtol=2e-15, atol=0)


def test_rbend_length_parameter_updates():
    env = xt.Environment()
    env['a'] = 0.2
    env['diff'] = 0.
    env['ls'] = 0.4713
    env.new('b', xt.RBend, length_straight='ls', angle='a', rbend_angle_diff='diff')
    bend = env.get('b')
    env['a'] = 0.
    env['diff'] = np.deg2rad(22.)
    env['ls'] = 0.8
    expected_length = 0.8 / np.cos(np.deg2rad(11.))
    np.testing.assert_allclose(bend.length, expected_length, rtol=2e-15, atol=0)
    assert bend.h == 0.
    assert bend._k0 == 0.

    env['a'] = -1e-12
    np.testing.assert_allclose(bend.h, -1e-12 / expected_length, rtol=2e-15, atol=0)
    assert bend._k0 == bend.h
    bend.k0 = 0.123
    env['a'] = 1e-12
    assert bend.k0 == 0.123
    np.testing.assert_allclose(bend.h, 1e-12 / expected_length, rtol=2e-15, atol=0)

    restored = xt.RBend.from_dict(bend.to_dict())
    np.testing.assert_allclose(restored.length, expected_length, rtol=2e-15, atol=0)
    assert restored.k0 == 0.123


@pytest.mark.parametrize('angle', [0., 1e-12, 0.2])
def test_rbend_zero_length(angle):
    with np.errstate(divide='raise', invalid='raise'):
        bend = xt.RBend(length_straight=0., angle=angle, rbend_angle_diff=0.3)
    assert bend.length == 0.
    assert bend.h == 0.
    assert bend._k0 == 0.
