# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

import numpy as np
import pytest
from scipy.constants import c
from xobjects.test_helpers import allow_kernel_compilation, for_all_test_contexts
import xtrack as xt


# Placed feedback uses the NumPy and CuPy numerical backends.
@for_all_test_contexts(excluding=('ContextPyopencl',))
@allow_kernel_compilation
def test_orbit_feedback_delivers_the_expected_particle_kick(
    test_context, sandbox_cwd,
):
    """Match the physical kick and BPM trace to a delayed, damped map."""
    reference = xt.Particles(_context=test_context, p0c=7e9)
    turns, first, interval, delay = 256, 16, 8, 3
    beta_y, tune, damping = 10.0, 0.25, 0.1
    length, disturbance = 1000.0, 1.5e-6
    period = length / (float(reference.beta0[0]) * c)

    phase = 2 * np.pi * tune
    transport = (1 - damping) * np.array([
        [np.cos(phase), beta_y * np.sin(phase)],
        [-np.sin(phase) / beta_y, np.cos(phase)],
    ])
    response = np.linalg.solve(np.eye(2) - transport, transport[:, 1])[0]
    gain = 0.35 / response

    env = xt.Environment(particle_ref=reference)
    env['trim_rad'] = 0.0
    env.elements['disturbance'] = xt.Multipole(ksl=[disturbance])
    env.new('corrector', xt.Multipole, ksl=['trim_rad'])
    env.elements['ring'] = xt.LineSegmentMap(
        length=length, qy=tune, bety=beta_y,
        damping_rate_y=damping, damping_rate_py=damping,
    )
    for name, quantity in [('before', 'mean_py'), ('after', 'mean_py'),
                           ('pickup', 'mean_y')]:
        env.elements[name] = xt.BeamStatsMonitor(
            _context=test_context, stop_at_turn=turns, stats=[quantity],
        )

    orbit = xt.feedback.Feedback(
        name='orbit', update_turns=interval, first_calculation_turn=first,
        command_delay_seconds=(delay - 0.5) * period,
    )
    orbit.inputs.new(name='y', monitor='pickup', quantity='mean_y')
    orbit.outputs.new(name='trim', target='trim_rad')
    orbit.parameters.new(name='gain', value=gain)
    orbit.state.new(name='trim', initial_value=0.0)
    orbit.program = '''
state.trim = state.trim - parameters.gain * acquired.y
outputs.trim = state.trim
'''
    env.elements['orbit'] = orbit
    line = env.new_line(components=[
        'disturbance', 'before', 'corrector', 'after', 'ring', 'pickup', 'orbit',
    ])
    line_off = line.copy(_context=test_context)
    for tracked_line, active in [(line_off, False), (line, True)]:
        tracked_line.build_tracker(_context=test_context)
        tracked_line.feedback.enabled = active
        tracked_line.track(reference.copy(), num_turns=turns)

        # The applied command changes the transported coordinates on the
        # traversal after delivery. The completed-turn number is its index.
        expected_y = np.zeros(turns)
        expected_trim = np.zeros(turns)
        state = np.zeros(2)
        request = 0.0
        for i in range(turns):
            state = transport @ (state + [0.0, disturbance + expected_trim[i]])
            expected_y[i] = state[0]
            completed = i + 1
            if (active and completed >= first
                    and (completed - first) % interval == 0):
                request -= gain * state[0]
                expected_trim[completed + delay:] = request

        measured_trim = (
            tracked_line.get('after').mean_py - tracked_line.get('before').mean_py
        )
        np.testing.assert_allclose(
            measured_trim, expected_trim, rtol=1e-12, atol=1e-18,
        )
        np.testing.assert_allclose(
            tracked_line.get('pickup').mean_y, expected_y, rtol=1e-12, atol=1e-17,
        )

    assert line_off.get('pickup').mean_y[-1] == pytest.approx(
        response * disturbance, rel=1e-6,
    )
    residual = np.max(np.abs(line.get('pickup').mean_y[-16:]))
    assert residual < 0.01 * response * disturbance
    assert line['trim_rad'] == pytest.approx(-disturbance, rel=0.01)


@for_all_test_contexts(excluding=('ContextPyopencl',))
@allow_kernel_compilation
def test_two_correctors_restore_the_closed_orbit_at_two_pickups(
    test_context, sandbox_cwd,
):
    """Recover two signed corrector kicks from an independent closed orbit."""
    reference = xt.Particles(_context=test_context, p0c=7e9)
    beta_y, tune, decay = 8.0, 0.31, 0.9
    interval, turns, error = 32, 192, 2e-6
    phase = 2 * np.pi * tune / 4
    transport = decay * np.array([
        [np.cos(phase), beta_y * np.sin(phase)],
        [-np.sin(phase) / beta_y, np.cos(phase)],
    ])
    powers = [np.linalg.matrix_power(transport, i) for i in range(5)]

    def closed_orbit(kicks):
        forced = sum(powers[4 - i][:, 1] * kick for i, kick in enumerate(kicks))
        start = np.linalg.solve(np.eye(2) - powers[4], forced)
        return [
            (powers[j] @ start)[0] + sum(
                powers[j - i][0, 1] * kicks[i] for i in range(j)
            )
            for j in [1, 3]
        ]

    response = np.column_stack((
        closed_orbit([1, 0, 0, 0]), closed_orbit([0, 0, 1, 0]),
    ))
    displaced = closed_orbit([0, error, 0, 0])
    expected_trim = np.linalg.solve(response, -np.array(displaced))

    env = xt.Environment(particle_ref=reference)
    for i in range(4):
        env.elements[f'arc_{i}'] = xt.LineSegmentMap(
            length=250.0, qy=tune / 4, bety=beta_y,
            damping_rate_y=1 - decay, damping_rate_py=1 - decay,
        )
    for i in [0, 1]:
        env[f'trim_{i}'] = 0.0
        env.new(f'corrector_{i}', xt.Multipole, ksl=[f'trim_{i}'])
        env.elements[f'bpm_{i}'] = xt.BeamStatsMonitor(
            _context=test_context, stop_at_turn=turns, stats=['mean_y'],
        )
    env.new('error', xt.Multipole, ksl=[error])
    fb = xt.feedback.Feedback(name='orbit', update_turns=interval)
    for i in [0, 1]:
        fb.inputs.new(name=f'y{i}', monitor=f'bpm_{i}', quantity='mean_y')
    fb.outputs.new(name='trim', target=['trim_0', 'trim_1'])
    fb.state.new(name='trim', initial_value=np.zeros(2))
    fb.parameters.new(name='response', value=response)
    fb.program = '''
error = stack((acquired.y0, acquired.y1))
state.trim = state.trim - linalg.solve(parameters.response, error)
outputs.trim = state.trim
'''
    env.elements['orbit'] = fb
    line = env.new_line(components=[
        'corrector_0', 'arc_0', 'bpm_0', 'error', 'arc_1',
        'corrector_1', 'arc_2', 'bpm_1', 'arc_3', 'orbit',
    ])

    line.build_tracker(_context=test_context)
    line.feedback.enable()
    line.track(reference.copy(), num_turns=turns)

    traces = np.column_stack([line.get(f'bpm_{i}').mean_y for i in [0, 1]])
    np.testing.assert_allclose(traces[interval - 1], displaced, atol=1e-10)
    np.testing.assert_allclose(traces[-16:], 0, atol=1e-12)
    np.testing.assert_allclose(
        [line.get(f'corrector_{i}').ksl[0] for i in [0, 1]],
        expected_trim, atol=1e-12,
    )
