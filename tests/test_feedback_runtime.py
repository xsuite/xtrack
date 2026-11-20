# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

"""Transactional runtime semantics independent of lattice/JIT fixtures."""

from types import SimpleNamespace

import numpy as np
import pytest
import xobjects as xo
import xtrack as xt
from xobjects.test_helpers import for_all_test_contexts

from xtrack.feedback import Acquisition, KnobOutput
from xtrack.feedback.runtime import FeedbackController, FeedbackSystem


class Source:
    def __init__(self, values, context):
        self.context = context
        self.values = values

    def read_record(self, turn, *, completed_turn):
        assert turn < completed_turn
        xp = self.context.nplike_lib
        value = xp.asarray(self.values[turn])
        return SimpleNamespace(value=value, valid=xp.isfinite(value))


class Output:
    def __init__(self, name, values=None):
        self.name = name
        self.values = np.array([0.0] if values is None else values)
        self.ownership_keys = (('fake', name),)
        self.dependent_field_keys = ()

    def snapshot(self):
        return self.values.copy()

    _checkpoint = snapshot

    def _restore_checkpoint(self, checkpoint):
        self.values = checkpoint

    def validate(self, values):
        values = np.asarray(values, dtype=float).reshape(-1)
        if values.shape != self.values.shape or not np.all(np.isfinite(values)):
            raise ValueError('Invalid command')
        return values.copy()

    def apply(self, values):
        values = self.validate(values)
        self.values = values


def make_controller(values, program, *, name='controller', memory=None, **kwargs):
    source = Source(values, xo.ContextCpu())
    output = Output(name)
    controller = FeedbackController(
        name,
        sources={'y': source},
        program=program,
        output=output,
        memory=memory,
        **kwargs,
    )
    return controller, output


def test_held_acquisition_does_not_duplicate_controller_history():
    """Only new publications enter the history, including on skipped turns."""
    observed = []

    def program(inputs, event):
        observed.append((inputs['y_history'].copy(), event['acquisition_published']))
        return {'outputs': {'command': [inputs['y']]}}

    controller, _ = make_controller(
        [1.0, 99.0, 3.0, 99.0, 5.0],
        program,
        history_length=2,
        acquisition=Acquisition(sample_turns=2),
    )
    system = FeedbackSystem([controller])
    for turn in range(1, 6):
        system.step(turn, float(turn))
    assert [published for _, published in observed] == [True, False, True]
    for (history, _), expected in zip(observed, ([1.0, 3.0], [1.0, 3.0], [3.0, 5.0])):
        np.testing.assert_array_equal(history, expected)


def test_invalid_acquisition_clears_derived_controller_history():
    """Rebuild history from consecutive valid transformed measurements."""
    observed = []

    def transform(inputs):
        return {'position': inputs['y']}, True

    def program(inputs):
        observed.append(inputs['position_history'].copy())
        return {'outputs': {'command': [inputs['position']]}}

    controller, _ = make_controller(
        [1.0, 2.0, np.nan, 4.0, 5.0],
        program,
        history_length=2,
        acquisition=Acquisition(transform=transform),
    )
    system = FeedbackSystem([controller])
    for turn in range(1, 6):
        system.step(turn, float(turn))
    np.testing.assert_array_equal(observed, [[1.0, 2.0], [4.0, 5.0]])


def test_history_retry_after_callback_failure_does_not_duplicate_samples():
    """Accept each passage only once when retrying a stopped boundary."""
    fail = [True]

    def program(inputs, previous):
        if fail[0]:
            raise ValueError('not accepted')
        np.testing.assert_array_equal(inputs['y_history'], [1.0, 2.0])
        return {
            'memory': {'count': previous['count'] + 1},
            'outputs': {'command': [3.0]},
        }

    controller, output = make_controller(
        [1.0, 2.0], program, memory={'count': np.zeros(1)}, history_length=2
    )
    system = FeedbackSystem([controller])
    system.step(1, 1.0)
    with pytest.raises(ValueError):
        system.step(2, 2.0)
    assert len(controller._history['y']) == 1
    fail[0] = False
    system.step(2, 2.0)
    np.testing.assert_array_equal(controller.memory['count'], [1.0])
    np.testing.assert_array_equal(output.values, [3.0])
    with pytest.raises(ValueError):
        system.step(2, 2.0)


def test_delayed_delivery_and_response_advance_without_calculation():
    """Advance the held actuator on turns without controller calculations."""
    def program(previous):
        command = previous['command'] + 1
        return {'memory': {'command': command}, 'outputs': {'command': command}}

    controller, output = make_controller(
        [1.0] * 5,
        program,
        memory={'command': np.zeros(1)},
        update_turns=3,
        first_calculation_turn=1,
        command_delay_seconds=0.1,
        response_time_constant_seconds=1.0,
    )
    system = FeedbackSystem([controller])
    first = system.step(1, 1.0)[0]
    assert first['executed'] and len(controller.queue) == 1
    controller.disable()
    second = system.step(2, 2.0)[0]
    assert not second['executed'] and len(controller.queue) == 0
    assert output.values[0] == 0.0  # Finite response is continuous at delivery.
    third = system.step(3, 3.0)[0]
    assert not third['executed']
    assert output.values[0] == pytest.approx(1 - np.exp(-1))


def test_invalid_source_clears_fresh_history_but_queued_commands_deliver():
    """Deliver an already queued command while the pickup history recovers."""
    def program(inputs):
        return {'outputs': {'command': [float(np.sum(inputs['y_history']))]}}

    controller, output = make_controller(
        [1.0, 2.0, np.nan, 4.0, 5.0],
        program,
        history_length=2,
        command_delay_seconds=0.1,
    )
    system = FeedbackSystem([controller])
    system.step(1, 1.0)
    system.step(2, 2.0)
    invalid = system.step(3, 3.0)[0]
    assert not invalid['source_valid'] and not invalid['executed']
    assert output.values[0] == 3.0
    assert len(controller._history['y']) == 0
    assert not system.step(4, 4.0)[0]['executed']
    assert system.step(5, 5.0)[0]['executed']


@pytest.mark.parametrize('error_type', [RuntimeError, KeyboardInterrupt])
def test_failed_output_leaves_both_commands_and_controllers_retriable(
    monkeypatch, error_type
):
    """Restore both writes and retry the same completed passage."""
    def program(previous):
        return {
            'memory': {'count': previous['count'] + 1},
            'outputs': {'command': [1.0]},
        }

    first, out1 = make_controller(
        [1.0], program, name='first', memory={'count': np.zeros(1)}
    )
    second, out2 = make_controller(
        [1.0], program, name='second', memory={'count': np.zeros(1)}
    )
    system = FeedbackSystem([first, second])
    apply = out2.apply

    def write_then_fail(values):
        apply(values)
        if np.any(values):
            raise error_type('Output failed after its field was written')

    monkeypatch.setattr(out2, 'apply', write_then_fail)
    with pytest.raises(error_type):
        system.step(1, 1.0)
    assert out1.values[0] == out2.values[0] == 0.0
    assert first.memory['count'][0] == second.memory['count'][0] == 0.0
    monkeypatch.setattr(out2, 'apply', apply)
    system.step(1, 1.0)
    assert out1.values[0] == out2.values[0] == 1.0


def test_nonfinite_diagnostics_can_describe_a_valid_held_command():
    """Report an unavailable estimate while holding a finite command."""
    def program():
        return {
            'outputs': {'command': [0.0]},
            'diagnostics': {'accepted': False, 'estimate': np.nan},
        }

    controller, _ = make_controller([1.0], program)
    entry = FeedbackSystem([controller]).step(1, 1.0)[0]
    assert np.isnan(entry['diagnostics']['estimate'])
    assert not entry['diagnostics']['accepted']


def test_previous_memory_is_not_mutable_cpu_callback_input():
    """Reject an in-place write to the controller's previous CPU memory."""
    def program(previous):
        previous['count'][:] = 99.0
        return {'outputs': {'command': [0.0]}}

    controller, _ = make_controller([1.0], program, memory={'count': np.zeros(1)})
    with pytest.raises(ValueError):
        FeedbackSystem([controller]).step(1, 1.0)
    np.testing.assert_array_equal(controller.memory['count'], [0.0])


def test_reject_overlapping_fields_and_processing_reservations():
    """Reject two physical writers or overlapping jobs on one processor."""
    def program():
        return {'outputs': {'command': [0.0]}}

    first, out1 = make_controller(
        [1.0, 1.0], program, name='one', processing_delay_seconds=2.0
    )
    second, out2 = make_controller([1.0, 1.0], program, name='two')
    out1.dependent_field_keys = out2.dependent_field_keys = (('physical', 1, 'ksl', 0),)
    with pytest.raises(ValueError):
        FeedbackSystem([first, second])
    system = FeedbackSystem([first])
    system.step(1, 1.0)
    with pytest.raises(ValueError, match='overlaps'):
        system.step(2, 2.0)


def test_disabled_controller_holds_target_and_hardware_keeps_settling():
    """Pause calculations while preserving the last actuator target."""
    def program():
        return {'outputs': {'command': [1.0]}}

    controller, output = make_controller(
        [1.0] * 4, program, response_time_constant_seconds=1.0
    )
    system = FeedbackSystem([controller])
    system.step(1, 1.0)
    controller.disable()
    assert not system.step(2, 2.0)[0]['executed']
    assert output.values[0] == pytest.approx(1 - np.exp(-1))
    controller.enable()
    assert system.step(3, 3.0)[0]['executed']


def test_cancelled_pending_request_requires_explicit_state_reconciliation():
    """Require a state decision when a command is cancelled before delivery."""
    def program(inputs, event, previous):
        assert inputs['applied_output'][0] == 0.0
        assert inputs['held_target'][0] == 0.0
        assert event['pending_count'] == 0
        command = previous['command'] + 1
        return {'memory': {'command': command}, 'outputs': {'command': command}}

    controller, _ = make_controller(
        [1.0, 1.0], program, memory={'command': np.zeros(1)}, command_delay_seconds=1.0
    )
    system = FeedbackSystem([controller])
    system.step(1, 1.0)
    controller.disable(cancel_pending=True)
    with pytest.raises(RuntimeError, match='reconciliation'):
        controller.enable()
    controller.enable(memory={'command': np.zeros(1)})
    system.step(2, 2.0)


def test_internal_commit_failure_is_fatal_and_cannot_double_commit_on_retry():
    """Restore both physical outputs and stop after an internal commit fails."""
    def program(previous):
        return {
            'memory': {'count': previous['count'] + 1},
            'outputs': {'command': [1.0]},
        }

    first, output1 = make_controller(
        [1.0], program, name='one', memory={'count': np.zeros(1)}
    )
    second, output2 = make_controller(
        [1.0], program, name='two', memory={'count': np.zeros(1)}
    )
    system = FeedbackSystem([first, second])
    original = second.actuator.commit

    def failed_commit(candidate):
        raise MemoryError('Simulated allocation failure')

    second.actuator.commit = failed_commit
    with pytest.raises(RuntimeError, match='cannot be retried'):
        system.step(1, 1.0)
    assert output1.values[0] == output2.values[0] == 0.0
    second.actuator.commit = original
    before = second.memory['count'].copy()
    with pytest.raises(RuntimeError, match='unusable'):
        system.step(1, 1.0)
    np.testing.assert_array_equal(second.memory['count'], before)


def test_monitor_source_requires_unique_participating_line():
    """Bind pickup storage to exactly one occurrence in an active tracker."""
    context = xo.ContextCpu()
    source = Source([1.0], context)
    monitor = SimpleNamespace(every_n_turns=1)
    source.monitor = monitor
    controller = FeedbackController(
        'bound',
        sources={'y': source},
        program=lambda: {'outputs': {'command': [0.0]}},
        output=Output('bound'),
    )
    with pytest.raises(ValueError, match='participating'):
        FeedbackSystem([controller])
    line = SimpleNamespace(
        element_names=['bpm'], tracker=object(), get=lambda name: object()
    )
    with pytest.raises(ValueError, match='occur once'):
        FeedbackSystem([controller], lines=[line])
    line.get = lambda name: monitor
    system = FeedbackSystem([controller], lines=[line])
    line.tracker = object()
    with pytest.raises(RuntimeError, match='binding changed'):
        system.step(1, 1.0)


@pytest.mark.parametrize('mutation', ['source_item', 'sources_mapping'])
def test_source_replacement_is_rejected_before_any_write(mutation):
    """Reject a changed source binding before publishing any command."""
    def program(inputs):
        return {'outputs': {'command': [inputs['y']]}}

    controller, output = make_controller([1.0], program)
    system = FeedbackSystem([controller])
    replacement = Source([2.0], controller.context)
    if mutation == 'source_item':
        controller.sources['y'] = replacement
    else:
        controller.sources = {'y': replacement}
    with pytest.raises(RuntimeError, match='Source bindings changed'):
        system.step(1, 0.1)
    np.testing.assert_array_equal(output.values, [0.0])
    assert not controller.log


def test_controller_ownership_and_activation_during_step():
    """Keep one controller owner and forbid reentrant activation changes."""
    def program():
        controller.disable()
        return {'outputs': {'command': [1e-6]}}

    controller, output = make_controller([1.0], program)
    system = FeedbackSystem([controller])
    with pytest.raises(ValueError, match='only one feedback system'):
        FeedbackSystem([controller])
    with pytest.raises(RuntimeError, match='between boundaries'):
        system.step(1, 1.0)
    assert controller.enabled
    assert output.values[0] == 0.0


def test_exact_field_rollback_when_another_output_fails(monkeypatch):
    """Recover a directly installed corrector field and its independent knob."""
    env = xt.Environment()
    env['trim'] = 0.0
    env.new('corrector', xt.Multipole, ksl=[0.0])
    env.ref['corrector'].ksl[0] = env.ref['trim']
    env.get('corrector').ksl[0] = 0.125
    real_output = KnobOutput(env, 'trim')
    context = xo.ContextCpu()
    first = FeedbackController(
        'first', sources={'y': Source([1.0], context)},
        program=lambda: {'outputs': {'command': [1e-6]}}, output=real_output,
    )
    second, output = make_controller(
        [1.0], lambda: {'outputs': {'command': [1e-6]}}, name='second',
    )
    system = FeedbackSystem([first, second])

    def fail(values):
        raise ValueError('Second output failed')

    with monkeypatch.context() as patch:
        patch.setattr(output, 'apply', fail)
        with pytest.raises(ValueError, match='Second output failed'):
            system.step(1, 1.0)
    assert env['trim'] == 0.0
    assert env.get('corrector').ksl[0] == 0.125
    system.step(1, 1.0)
    assert env.get('corrector').ksl[0] == 1e-6


def test_update_phase_is_independent_of_history(monkeypatch):
    """Run the first update on turn three and skip unchanged field writes."""
    controller, output = make_controller(
        [1.0] * 6, lambda: {'outputs': {'command': [1e-6]}},
        update_turns=3, history_length=2,
    )
    writes = []
    apply = output.apply

    def record_write(values):
        writes.append(values[0])
        apply(values)

    monkeypatch.setattr(output, 'apply', record_write)
    system = FeedbackSystem([controller])
    entries = [system.step(turn, float(turn))[0] for turn in range(1, 7)]
    assert [row['completed_turn'] for row in entries if row['executed']] == [3, 6]
    assert writes == [1e-6]


@pytest.mark.parametrize('cadence', ['turns', 'seconds'])
@pytest.mark.parametrize('explicit_first', [False, True])
@pytest.mark.parametrize('fixed_clock', [False, True])
def test_update_grids_at_a_nonzero_epoch(cadence, explicit_first, fixed_clock):
    """Schedule from the run start and preserve explicit absolute phases."""
    start_turn, epoch, period = 100, 1e6, 1e-6
    if cadence == 'turns':
        settings = dict(update_turns=3)
        if explicit_first:
            settings['first_calculation_turn'] = start_turn + 2
    else:
        settings = dict(update_frequency_hz=1 / (3 * period))
        if explicit_first:
            settings['first_calculation_time_seconds'] = epoch + 2 * period
    controller, _ = make_controller(
        [1.0] * 107, lambda: {'outputs': {'command': [0.0]}},
        initial_time_seconds=0.0 if fixed_clock else epoch, **settings,
    )
    system = FeedbackSystem(
        [controller], _turn_clock=(start_turn, epoch, period) if fixed_clock else None,
    )
    entries = []
    for offset in range(1, 7):
        entry, = system.step(start_turn + offset, epoch + offset * period)
        if entry['executed']:
            entries.append(entry)
    first = 2 if explicit_first else 3
    assert [row['completed_turn'] for row in entries] == [
        start_turn + first, start_turn + first + 3,
    ]
    assert [row['missed_deadlines'] for row in entries] == [0, 0]
    assert entries[0]['dt_seconds'] == 0.0
    tolerance = 1e-17 if fixed_clock else np.spacing(epoch)
    assert entries[1]['dt_seconds'] == pytest.approx(3 * period, rel=0, abs=tolerance)
    if cadence == 'turns':
        assert entries[0]['nominal_reference_turns'] == start_turn + first
    else:
        assert entries[0]['nominal_time_seconds'] == pytest.approx(
            epoch + first * period, rel=0, abs=np.spacing(epoch))


def test_external_applied_value_change_is_rejected_before_calculation():
    """Reject an applied change before a calculation can use old readbacks."""
    controller, output = make_controller(
        [], lambda: {'outputs': {'command': [1e-6]}},
    )
    system = FeedbackSystem([controller])
    output.values[0] = 2e-6
    with pytest.raises(RuntimeError, match='outside its response owner'):
        system.step(1, 1.0)
    assert output.values[0] == 2e-6
    assert not controller.log


@for_all_test_contexts(excluding=('ContextPyopencl',))
def test_callback_configuration_is_owned_at_each_calculation(test_context):
    """A callback's local writes cannot change the stored settings."""
    xp = test_context.nplike_lib

    def program(parameters, constants):
        command = parameters['gain'] + constants['offset']
        for array in (parameters['gain'], constants['offset']):
            if isinstance(array, np.ndarray):
                array.flags.writeable = True
            array[:] = 99.0
        return {'outputs': {'command': command}}

    env = xt.Environment()
    env['trim'] = 0.0
    controller = FeedbackController(
        'trim', sources={'y': Source([1.0, 1.0], test_context)}, program=program,
        output=KnobOutput(env, 'trim'),
        parameters={'gain': xp.array([2e-6])},
        constants={'offset': xp.array([3e-6])},
    )
    system = FeedbackSystem([controller])
    for turn in (1, 2):
        system.step(turn, float(turn))
        assert controller.output.read()[0] == pytest.approx(5e-6)
