# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

import numpy as np
import pytest
import xtrack as xt
from xobjects.test_helpers import (
    allow_kernel_compilation, for_all_test_contexts,
)


# Placed feedback uses the NumPy and CuPy numerical backends.
@for_all_test_contexts(excluding=('ContextPyopencl',))
@allow_kernel_compilation
def test_orbit_and_tune_feedback_use_distinct_cadences_on_one_line(
    test_context, sandbox_cwd,
):
    """Measure tune from phase advance while an independent orbit loop runs."""
    bare_tune, target_tune = 0.28, 0.31
    beta_x, beta_y, update_turns, turns = 10.0, 8.0, 16, 160
    disturbance, decay = 1e-6, 0.9
    reference = xt.Particles(_context=test_context, p0c=7e9)
    env = xt.Environment(particle_ref=reference)
    env['qx_trim'] = env['orbit_trim'] = 0.0
    env.new('error', xt.Multipole, ksl=[disturbance])
    env.new('corrector', xt.Multipole, ksl=['orbit_trim'])
    env.elements['arc'] = xt.LineSegmentMap(
        length=1000.0, betx=beta_x, qx=bare_tune, bety=beta_y, qy=0.25,
        damping_rate_y=1 - decay, damping_rate_py=1 - decay,
    )
    env.ref['arc'].qx = bare_tune + env.ref['qx_trim']
    env.elements['bpm'] = xt.BeamStatsMonitor(
        _context=test_context, stop_at_turn=turns,
        stats=['mean_x', 'mean_px', 'mean_y'],
    )
    tune = xt.feedback.Feedback(
        name='tune', history_length=update_turns, update_turns=update_turns,
    )
    tune.inputs.new(name='x', monitor='bpm', quantity='mean_x')
    tune.outputs.new(name='trim', target='qx_trim')
    tune.parameters.new(name='target', value=np.cos(2 * np.pi * target_tune))
    tune.parameters.new(
        name='slope', value=-2 * np.pi * np.sin(2 * np.pi * target_tune),
    )
    tune.state.new(name='trim', initial_value=0.0)
    tune.program = '''
x = acquired.x_history
middle = x[1:-1]
cosine = (middle @ (x[:-2] + x[2:])) / (2 * (middle @ middle))
state.trim = state.trim + (parameters.target - cosine) / parameters.slope
outputs.trim = state.trim
'''
    orbit = xt.feedback.Feedback(
        name='orbit', update_turns=4, first_calculation_turn=16,
    )
    orbit.inputs.new(name='y', monitor='bpm', quantity='mean_y')
    orbit.outputs.new(name='kick', target='orbit_trim')
    orbit.state.new(name='kick', initial_value=0.0)
    orbit.program = '''
state.kick = state.kick - 0.05 * acquired.y
outputs.kick = state.kick
'''
    env.elements['tune'] = tune
    env.elements['orbit'] = orbit
    line = env.new_line(components=[
        'bpm', 'error', 'corrector', 'arc', 'tune', 'orbit',
    ])
    particles = line.particle_ref.copy()
    particles.x[:] = 10e-6

    line.build_tracker(_context=test_context)
    line.feedback.enable()
    line.track(particles, num_turns=turns)

    # Use position and angle to measure the actual phase advance. The feedback
    # independently estimates the tune using the BPM position alone.
    bpm = line.get('bpm')
    phase = bpm.mean_x - 1j * beta_x * bpm.mean_px
    witness = np.angle(phase[1:] * phase[:-1].conj()) / (2 * np.pi)
    np.testing.assert_allclose(witness[:update_turns], bare_tune, atol=1e-12)
    assert witness[update_turns] > bare_tune
    np.testing.assert_allclose(witness[-update_turns + 1:], target_tune, atol=5e-7)
    bare_orbit = beta_y * decay * disturbance / (1 + decay**2)
    assert np.max(np.abs(bpm.mean_y[-16:])) < 0.05 * bare_orbit
    assert line.get('corrector').ksl[0] == pytest.approx(-disturbance, rel=0.05)
    assert [row['completed_turn'] for row in orbit.log if row['executed']] == list(
        range(16, turns + 1, 4)
    )
    assert [row['completed_turn'] for row in tune.log if row['executed']] == list(
        range(16, turns + 1, 16)
    )
