# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

import numpy as np
import pytest
from numpy.testing import assert_allclose
from xobjects.test_helpers import for_all_test_contexts

from xtrack.feedback.program import _Program
from xtrack.feedback.timing import _CalculationTiming


@for_all_test_contexts(excluding=('ContextPyopencl',))
def test_feedback_program_stages_pi_correction(test_context):
    """Compute the PI correction without changing the previous integral."""
    xp = test_context.nplike_lib
    to_host = test_context.nparray_from_context_array
    program = _Program(
        'error = acquired.y - parameters.reference\n'
        'state.integral = state.integral - parameters.ki * event.dt_seconds * error\n'
        'outputs.kick = -parameters.kp * error + state.integral',
        name='orbit', acquired=['y'], parameters=['reference', 'ki', 'kp'],
        state=['integral'], output='kick', xp=xp,
    )
    previous = {'integral': xp.zeros(2)}
    event = _CalculationTiming(dt_seconds=0.25, time_seconds=1.0, completed_turn=4)
    result = program({'y': xp.array([1.0, -2.0])},
                     {'reference': 0.0, 'ki': 0.5, 'kp': 0.1}, previous, event)
    assert isinstance(result['outputs']['kick'], xp.ndarray)
    assert_allclose(to_host(result['outputs']['kick']), [-0.225, 0.45])
    assert_allclose(to_host(result['memory']['integral']), [-0.125, 0.25])
    assert_allclose(to_host(previous['integral']), [0.0, 0.0])


@for_all_test_contexts(excluding=('ContextPyopencl',))
def test_feedback_program_binds_native_solver(test_context):
    """Apply a calibrated response inversion on the tracking context."""
    xp = test_context.nplike_lib
    program = _Program(
        'mean_y = mean(acquired.y_history, axis=0)\n'
        'outputs.kick = -linalg.solve(parameters.response, mean_y)',
        name='two_bpms', acquired=['y_history'], parameters=['response'],
        state=[], output='kick', xp=xp,
    )
    inputs = {'y_history': xp.array([[1.0, 5.0], [7.0, 5.0]])}
    parameters = {'response': xp.array([[2.0, 1.0], [1.0, 2.0]])}
    event = _CalculationTiming(dt_seconds=0.0, time_seconds=1.0, completed_turn=1)
    result = program(inputs, parameters, {}, event)
    assert isinstance(result['outputs']['kick'], xp.ndarray)
    assert_allclose(test_context.nparray_from_context_array(result['outputs']['kick']),
                    [-1.0, -2.0])


@for_all_test_contexts(excluding=('ContextPyopencl',))
def test_feedback_program_combines_scalar_settings_and_context_arrays(test_context):
    """Drive one corrector and hold the other at zero on the same context."""
    xp = test_context.nplike_lib
    program = _Program(
        'kick = -mean(parameters.gain) * acquired.y[0]\n'
        'outputs.kick = stack((where(parameters.enabled, kick, 0.0), 0.0))',
        name='two_correctors', acquired=['y'], parameters=['gain', 'enabled'],
        state=[], output='kick', xp=xp,
    )
    event = _CalculationTiming(dt_seconds=0.0, time_seconds=1.0, completed_turn=1)
    result = program({'y': xp.array([2e-6])},
                     {'gain': 0.5, 'enabled': True}, {}, event)
    kick = result['outputs']['kick']
    assert isinstance(kick, xp.ndarray)
    assert_allclose(test_context.nparray_from_context_array(kick), [-1e-6, 0.0])


@pytest.mark.parametrize('source, message', [
    ('import numpy', 'Only numerical assignments'),
    ('outputs.kick = np.sin(acquired.y)', 'Unsupported operation'),
    ('operation = sin\noutputs.kick = operation(acquired.y)', 'Unknown local'),
    ('outputs.kick = acquired.y.get()', 'Unsupported operation'),
    ('outputs.kick = linalg.eigvals(acquired.y)', 'Unsupported operation'),
    ('outputs.kick = sin(acquired.y, acquired.y)', 'positional arguments'),
    ('outputs.kick = sum(acquired.y, out=acquired.y)', 'literal axis'),
    ('outputs.kick = sum(acquired.y, axis=acquired.y[0])', 'literal axis'),
    ('outputs.kick = acquired.y[acquired.y[0]]', 'integer literals'),
    ('outputs.kick = acquired.y[::0]', 'Slice step must be nonzero'),
    ('outputs.kick = stack(())', 'stack requires at least one value'),
    ('outputs.kick = [acquired.y[0], acquired.y[1]]', 'Unsupported expression'),
    ('acquired.y[0] = 1.0', 'bound numerical name'),
    ('parameters.gain = 1.0', 'bound numerical name'),
    ('outputs.kick = acquired.__class__', 'bound numerical name'),
    ('outputs.kick = acquired.absent', 'Unknown acquired field'),
    ('state.absent = acquired.y', 'Unknown state field'),
    ('outputs.other = acquired.y', 'Unknown outputs field'),
    ('outputs.kick = event.pending_count', 'Unknown event field'),
    ('state.x = acquired.y', 'Assign the bound output'),
])
def test_feedback_program_rejects_invalid_blocks_at_binding(source, message):
    """Reject unsupported calls, writes and namespace fields before tracking."""
    with pytest.raises(ValueError, match=f"Feedback 'test', program line .*{message}"):
        _Program(source, name='test', acquired=['y'], parameters=['gain'],
                 state=['x'], output='kick', xp=np)


@for_all_test_contexts(excluding=('ContextPyopencl',))
def test_feedback_program_reports_solver_failure_on_its_source_line(test_context):
    """Leave the previous state intact and identify a failed native operation."""
    xp = test_context.nplike_lib
    program = _Program(
        'state.x = state.x + acquired.y\n'
        'outputs.kick = linalg.solve(parameters.response, state.x)',
        name='singular_response', acquired=['y'], parameters=['response'],
        state=['x'], output='kick', xp=xp,
    )
    previous = {'x': xp.zeros(2)}
    event = _CalculationTiming(dt_seconds=0.0, time_seconds=1.0, completed_turn=1)
    with pytest.raises(ValueError, match=r"singular_response'.*line 2:") as error:
        program({'y': xp.ones(2)}, {'response': xp.ones((2, 2))}, previous, event)
    assert isinstance(error.value.__cause__, np.linalg.LinAlgError)
    assert_allclose(test_context.nparray_from_context_array(previous['x']), [0, 0])
