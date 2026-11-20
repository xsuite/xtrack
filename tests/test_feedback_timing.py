# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

import dataclasses

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from xobjects.test_helpers import for_all_test_contexts

from xtrack.feedback.timing import (
    CommandQueue, DeadlineSchedule, FirstOrderActuator,
)


def test_turn_schedule_consumes_each_deadline_once():
    """Keep a turn deadline available until its update commits."""
    schedule = DeadlineSchedule(update_turns=2)
    assert schedule.poll(time_seconds=0.1, reference_turns=1) is None

    event = schedule.peek(time_seconds=0.2, reference_turns=2)
    assert event.dt_seconds == 0.0
    assert event.sequence_index == 0
    assert event.nominal_reference_turns == 2
    assert schedule.peek(time_seconds=0.2, reference_turns=2) == event
    assert schedule.next_deadline == 2

    schedule.commit(event)
    assert schedule.poll(time_seconds=0.2, reference_turns=2) is None
    assert schedule.poll(time_seconds=0.3, reference_turns=3) is None
    second = schedule.poll(time_seconds=0.4, reference_turns=4)
    assert second.sequence_index == 1
    assert second.dt_seconds == pytest.approx(0.2)
    assert schedule.next_deadline == 6
    with pytest.raises(ValueError, match='Stale'):
        schedule.commit(second)
    with pytest.raises(ValueError, match='backwards'):
        schedule.poll(time_seconds=0.3, reference_turns=3)


def test_frequency_schedule_keeps_nominal_phase():
    """Keep nominal phase when execution is rounded to a turn boundary."""
    revolution_frequency = 3300.0
    update_period = 0.012  # s
    first_time = 0.002  # s
    schedule = DeadlineSchedule(
        update_frequency_hz=1 / update_period, first_time=first_time)

    execution_turns = []
    deadlines = []
    for turn in range(1, 200):
        event = schedule.poll(time_seconds=turn / revolution_frequency,
                              reference_turns=turn)
        if event is not None:
            execution_turns.append(turn)
            deadlines.append(event.nominal_time_seconds)

    # The first deadline is at turn 6.6. Subsequent ones are 39.6 turns apart.
    expected_deadlines = first_time + np.arange(5) * update_period
    assert_allclose(deadlines, expected_deadlines, rtol=0, atol=1e-16)
    assert_array_equal(execution_turns, [7, 47, 86, 126, 165])


def test_frequency_schedule_counts_missed_deadlines():
    """Report or reject missed updates while leaving the nominal grid intact."""
    schedule = DeadlineSchedule(
        update_frequency_hz=100.0, missed_policy='coalesce')
    event = schedule.poll(time_seconds=0.035, reference_turns=1)
    assert event.missed_deadlines == 2
    assert event.nominal_time_seconds == 0.01
    assert schedule.next_deadline == 0.04

    strict = DeadlineSchedule(update_frequency_hz=100.0, missed_policy='raise')
    with pytest.raises(ValueError, match='missed 2'):
        strict.peek(time_seconds=0.035, reference_turns=1)
    assert strict.next_deadline == 0.01


@pytest.mark.parametrize(
    'kwargs',
    [
        {},
        dict(update_turns=1, update_frequency_hz=10),
        dict(update_turns=0),
        dict(update_turns=True),
        dict(update_turns=1.5),
        dict(update_frequency_hz=0),
        dict(update_frequency_hz=np.inf),
        dict(update_frequency_hz=np.nextafter(0.0, 1.0)),
        dict(update_turns=1, first_time=0),
        dict(update_frequency_hz=1, first_turn=0),
        dict(update_turns=1, missed_policy='silent'),
    ],
)
def test_schedule_rejects_invalid_configuration(kwargs):
    """Reject missing, ambiguous or invalid cadence configuration."""
    with pytest.raises(ValueError):
        DeadlineSchedule(**kwargs)


def test_schedule_rejects_deadlines_below_clock_resolution():
    """Reject a cadence whose successive deadlines collapse to one timestamp."""
    schedule = DeadlineSchedule(update_frequency_hz=1e18, first_time=1.0)
    with pytest.raises(ValueError, match='clock resolution'):
        schedule.peek(time_seconds=1.0, reference_turns=1)


def test_turn_schedule_keeps_a_future_fractional_deadline():
    """Wait for the next boundary when the nominal turn is fractional."""
    turn = 1e12
    schedule = DeadlineSchedule(update_turns=1, first_turn=turn + 0.01)
    assert schedule.poll(time_seconds=0.0, reference_turns=turn) is None
    assert schedule.poll(time_seconds=1.0, reference_turns=turn + 1) is not None


@for_all_test_contexts
def test_command_queue_owns_context_arrays(test_context):
    """Keep queued values on their context and isolate them from callers."""
    to_host = test_context.nparray_from_context_array
    initial = test_context.nparray_to_context_array(np.array([1.0, 2.0]))
    queue = CommandQueue()
    immediate = queue.enqueue(initial, time_seconds=0.0)
    delayed = queue.enqueue(
        initial, time_seconds=0.0, command_delay_seconds=0.1)

    initial.fill(999)
    queue.pending[0].value.fill(-999)
    ready = queue.peek_ready(0.0)
    assert [command.sequence_index for command in ready] == [immediate]
    assert isinstance(ready[0].value, type(initial))
    assert_array_equal(to_host(ready[0].value), [1.0, 2.0])
    assert len(queue) == 2

    ready[0].value.fill(0)
    assert_array_equal(to_host(queue.peek_ready(0.0)[0].value), [1.0, 2.0])
    queue.commit_ready(ready, time_seconds=0.0)
    assert queue.ready(0.0) == ()
    delivered = queue.ready(1.0)
    assert [command.sequence_index for command in delivered] == [delayed]
    assert delivered[0].ready_time_seconds == 0.1
    assert_array_equal(to_host(delivered[0].value), [1.0, 2.0])
    assert len(queue) == 0


@for_all_test_contexts
def test_command_queue_delivers_in_order_from_a_copy(test_context):
    """Order commands by arrival and preserve the original queue on preview."""
    to_host = test_context.nparray_from_context_array
    to_context = test_context.nparray_to_context_array
    values = [to_context(np.array([i])) for i in (1.0, 2.0, 3.0, 4.0)]
    queue = CommandQueue()
    late = queue.enqueue(values[0], time_seconds=0.0, command_delay_seconds=2.0)
    early = queue.enqueue(
        values[1], time_seconds=0.0, command_delay_seconds=1.0)
    tied = queue.enqueue(
        values[2], time_seconds=0.0, processing_delay_seconds=0.2,
        command_delay_seconds=0.8,
    )

    candidate = queue.copy()
    candidate.pending[0].value.fill(999)
    immediate = candidate.enqueue(values[3], time_seconds=0.0)
    due = candidate.ready(1.0)
    assert [cmd.sequence_index for cmd in due] == [immediate, early, tied]
    assert_array_equal(to_host(due[1].value), [2.0])

    assert len(queue) == 3
    assert_array_equal(to_host(queue.pending[0].value), [2.0])
    due = queue.ready(2.0)
    assert [cmd.sequence_index for cmd in due] == [early, tied, late]
    assert queue.enqueue(values[3], time_seconds=2.0) == immediate
    with pytest.raises(ValueError, match='backdate'):
        queue.enqueue(values[3], time_seconds=1.0)
    with pytest.raises(ValueError, match='backwards'):
        queue.ready(1.0)


def test_command_queue_rejects_stale_preview():
    """Keep commands pending when delivery used an outdated preview."""
    queue = CommandQueue()
    queue.enqueue([1.0], time_seconds=0.0)
    preview = queue.peek_ready(0.0)
    queue.enqueue([2.0], time_seconds=0.0)
    with pytest.raises(ValueError, match='Stale'):
        queue.commit_ready(preview, time_seconds=0.0)
    assert len(queue) == 2


def test_positive_delay_requires_a_later_boundary():
    """Delay a command even when its due time rounds to its request time."""
    queue = CommandQueue()
    queue.enqueue([1.0], time_seconds=1e10, command_delay_seconds=1e-10)
    assert queue.peek_ready(1e10) == ()
    assert len(queue.ready(1e10 + 1.0)) == 1


def test_zero_delay_command_waits_for_its_request():
    """Enforce request order even for adjacent floating-point timestamps."""
    request_time = np.nextafter(1.0, 2.0)
    queue = CommandQueue()
    queue.enqueue([1.0], time_seconds=request_time)
    assert queue.ready(1.0) == ()
    assert len(queue.ready(request_time)) == 1


@pytest.mark.parametrize('requested_turn', [40, 4000, 40000])
def test_reconstructed_turn_times_keep_exact_turn_delay(requested_turn):
    """Deliver a 33-turn command at the same relative turn for every request."""
    period = 1 / 3300.0
    queue = CommandQueue()
    queue.enqueue(
        [1.0],
        time_seconds=requested_turn * period,
        processing_delay_seconds=16 * period,
        command_delay_seconds=17 * period,
    )
    assert not queue.ready((requested_turn + 32) * period)
    assert len(queue.ready((requested_turn + 33) * period)) == 1


@pytest.mark.parametrize('offset', [-1e-8, 0.0, 1e-8])
def test_deadline_order_near_boundary_at_large_epoch(offset):
    """Resolve the sign of a 10 ns offset around a boundary at 10⁶ s."""
    epoch = 1e6
    period = 1e-5
    delay = period + offset
    deadline = epoch + delay
    schedule = DeadlineSchedule(
        update_frequency_hz=1 / period, first_time=deadline
    )
    queue = CommandQueue()
    queue.enqueue([1.0], time_seconds=epoch, command_delay_seconds=delay)

    boundary = epoch + period
    event = schedule.poll(time_seconds=boundary, reference_turns=1)
    delivered = queue.ready(boundary)
    assert (event is not None) == (offset <= 0.0)
    assert bool(delivered) == (offset <= 0.0)
    if offset > 0.0:
        boundary = epoch + 2 * period
        event = schedule.poll(time_seconds=boundary, reference_turns=2)
        assert event is not None
        assert len(queue.ready(boundary)) == 1


@pytest.mark.parametrize(
    'name,value',
    [
        ('time_seconds', -1.0),
        ('time_seconds', np.nan),
        ('processing_delay_seconds', -0.1),
        ('command_delay_seconds', np.inf),
    ],
)
def test_queue_rejects_invalid_time_configuration(name, value):
    """Reject negative and nonfinite timestamps and delays."""
    kwargs = dict(time_seconds=0.0)
    kwargs[name] = value
    with pytest.raises(ValueError):
        CommandQueue().enqueue([1.0], **kwargs)


# The actuator needs NumPy-style where and expm1 on context arrays.
@pytest.mark.parametrize('tau', [1.0, [1.0]])
@for_all_test_contexts(excluding=('ContextPyopencl',))
def test_actuator_resolves_short_response_intervals(test_context, tau):
    """Resolve field increments for intervals much shorter than tau."""
    xp = test_context.nplike_lib
    actuator = FirstOrderActuator(
        [0.0], response_time_constant_seconds=tau, xp=xp)
    actuator.set_target([1.0])

    applied = actuator.advance(1e-18)
    assert isinstance(applied, xp.ndarray)
    assert_allclose(
        test_context.nparray_from_context_array(applied), [1e-18],
        rtol=1e-15, atol=0,
    )


@pytest.mark.parametrize('tau', [1.0, [1.0]])
@for_all_test_contexts(excluding=('ContextPyopencl',))
def test_actuator_retains_a_small_settled_target(test_context, tau):
    """Resolve a small nonzero target after the initial field decays."""
    initial = 2e-3  # rad
    target = 1e-20  # rad
    time = 50.0  # s
    actuator = FirstOrderActuator(
        [initial], response_time_constant_seconds=tau, xp=test_context.nplike_lib)
    actuator.set_target([target])

    applied = actuator.advance(time)
    expected = target + (initial - target) * np.exp(-time)
    assert_allclose(test_context.nparray_from_context_array(applied), [expected],
                    rtol=1e-15, atol=0)


@for_all_test_contexts(excluding=('ContextPyopencl',))
def test_actuator_advances_toward_the_held_target(test_context):
    """Integrate the old target until delivery and commit the new target."""
    to_host = test_context.nparray_from_context_array
    actuator = FirstOrderActuator(
        [0.0], response_time_constant_seconds=0.005, xp=test_context.nplike_lib)

    # Deliver target at t=0: the physical state is continuous.
    initial = actuator.preview(0.0, target=[1.0])
    assert_array_equal(to_host(initial.applied), [0.0])
    actuator.commit(initial)

    candidate = actuator.preview(0.005, target=[2.0])
    assert_allclose(to_host(candidate.applied), [1 - np.exp(-1)])
    assert_array_equal(to_host(actuator.applied), [0.0])
    assert_array_equal(to_host(actuator.target), [1.0])

    actuator.commit(candidate)
    before = to_host(actuator.applied)
    assert_array_equal(to_host(actuator.advance(0.005)), before)
    assert_array_equal(to_host(actuator.target), [2.0])
    with pytest.raises(ValueError, match='Stale'):
        actuator.commit(candidate)
    with pytest.raises(ValueError, match='backwards'):
        actuator.preview(0.004)


@for_all_test_contexts(excluding=('ContextPyopencl',))
def test_coordinated_slew_preserves_a_linear_constraint(test_context):
    """Limit both moving channels together and leave the fixed channel alone."""
    to_host = test_context.nparray_from_context_array
    actuator = FirstOrderActuator(
        [0.0, 0.0, 0.0], maximum_slew_per_second=[1.0, 2.0, 0.0],
        xp=test_context.nplike_lib,
    )
    actuator.set_target([1.0, -1.0, 0.0])

    candidate = actuator.preview(0.1)
    assert_allclose(to_host(candidate.applied), [0.1, -0.1, 0.0])
    actuator.commit(candidate)
    assert_allclose(to_host(actuator.advance(0.2)), [0.2, -0.2, 0.0])


@pytest.mark.parametrize(
    'mode, immediate_step, common_fraction',
    [('independent', 1.0, 1.0), ('coordinated', 0.0, 0.1)],
)
@for_all_test_contexts(excluding=('ContextPyopencl',))
def test_slew_policy_controls_zero_tau_delivery(
    test_context, mode, immediate_step, common_fraction,
):
    """Apply an immediate step only when the channel can move independently."""
    to_host = test_context.nparray_from_context_array
    actuator = FirstOrderActuator(
        [0.0, 0.0, 0.0], response_time_constant_seconds=[0.0, 0.0, 0.1],
        maximum_slew_per_second=[np.inf, 1.0, np.inf], slew_mode=mode,
        xp=test_context.nplike_lib,
    )

    delivered = to_host(actuator.set_target([1.0, 1.0, 1.0]))
    advanced = to_host(actuator.advance(0.1))
    assert_array_equal(delivered, [immediate_step, 0.0, 0.0])
    assert_allclose(advanced, [
        common_fraction, 0.1, common_fraction * (1 - np.exp(-1)),
    ])


def test_actuator_rejects_invalid_candidates():
    """Reject incompatible shapes, stale candidates and foreign candidates."""
    actuator = FirstOrderActuator([0.0, 0.0])
    with pytest.raises(ValueError, match='shape'):
        actuator.preview(0.0, target=[1.0])
    candidate = actuator.preview(0.1)
    altered = dataclasses.replace(candidate, applied=np.zeros(3))
    with pytest.raises(ValueError, match='shape'):
        actuator.commit(altered)
    actuator.commit(candidate)
    with pytest.raises(ValueError, match='Stale'):
        actuator.commit(candidate)
    other = FirstOrderActuator([0.0, 0.0])
    with pytest.raises(ValueError, match='Stale'):
        other.commit(actuator.preview(0.2))


@for_all_test_contexts(excluding=('ContextPyopencl',))
def test_actuator_supports_distinct_time_constants(test_context):
    """Advance each channel by its own time constant, including zero."""
    to_host = test_context.nparray_from_context_array
    actuator = FirstOrderActuator(
        [1.0, -2.0, 0.5], response_time_constant_seconds=[0.0, 0.1, 0.4],
        xp=test_context.nplike_lib,
    )
    assert_array_equal(to_host(actuator.set_target([2, 3, -0.5])), [2, -2, 0.5])

    candidate = actuator.preview(0.1, target=[4.0, 9.0, 7.0])
    expected = [4.0, 3 - 5 * np.exp(-1), -0.5 + np.exp(-0.25)]
    assert_allclose(to_host(candidate.applied), expected)
    assert_array_equal(to_host(actuator.applied), [2, -2, 0.5])
    assert_array_equal(to_host(actuator.target), [2, 3, -0.5])

    actuator.commit(candidate)
    assert_allclose(
        to_host(actuator.advance(0.2)),
        [
            4,
            9 + (expected[1] - 9) * np.exp(-1),
            7 + (expected[2] - 7) * np.exp(-0.25),
        ],
    )


@for_all_test_contexts(excluding=('ContextPyopencl',))
def test_actuator_limits_each_sampled_increment(test_context):
    """Clip each channel's exponential increment to its own slew bound."""
    to_host = test_context.nparray_from_context_array
    actuator = FirstOrderActuator(
        [0, 0],
        response_time_constant_seconds=[0.1, 1.0],
        maximum_slew_per_second=[0.01, 100e-6],
        slew_mode='independent',
        xp=test_context.nplike_lib,
    )
    actuator.set_target([0.02, 200e-6])
    assert_allclose(to_host(actuator.advance(0.1)), [0.001, 10e-6])

    candidate = actuator.preview(0.2, target=[-0.01, -50e-6])
    assert_allclose(to_host(candidate.applied), [0.002, 20e-6])
    actuator.commit(candidate)

    expected = [0.001, -50e-6 + 70e-6 * np.exp(-0.1)]
    assert_allclose(to_host(actuator.advance(0.3)), expected)


@pytest.mark.parametrize('tau', [[-0.1, 1], [np.nan, 1], [np.inf, 1]])
def test_invalid_response_constants_are_rejected(tau):
    """Reject a negative or nonfinite time constant on any channel."""
    with pytest.raises(ValueError, match='finite and nonnegative'):
        FirstOrderActuator([0, 0], response_time_constant_seconds=tau)


@pytest.mark.parametrize('mode', ['coordinated', 'independent'])
@for_all_test_contexts(excluding=('ContextPyopencl',))
def test_actuator_response_is_independent_of_channel_units(test_context, mode):
    """Recover the same fields after rescaling channels and slew bounds."""
    xp = test_context.nplike_lib
    to_host = test_context.nparray_from_context_array
    scales = np.array([1e3, 1e6])  # rad -> mrad, m -> micrometres
    initial = np.array([0.005, -20e-6])
    rates = np.array([0.02, 100e-6])
    physical = FirstOrderActuator(
        initial,
        response_time_constant_seconds=[0.2, 0.7],
        maximum_slew_per_second=rates,
        slew_mode=mode,
        xp=xp,
    )
    converted = FirstOrderActuator(
        initial * scales,
        response_time_constant_seconds=[0.2, 0.7],
        maximum_slew_per_second=rates * scales,
        slew_mode=mode,
        xp=xp,
    )
    for time, target in [
        (0, [0.03, 80e-6]),
        (0.1, None),
        (0.2, [-0.01, -50e-6]),
        (0.5, None),
    ]:
        target = None if target is None else np.array(target)
        physical.commit(physical.preview(time, target=target))
        converted.commit(
            converted.preview(
                time,
                target=None if target is None else target * scales,
            )
        )
        assert_allclose(
            to_host(converted.applied) / scales, to_host(physical.applied),
            rtol=1e-13, atol=1e-18,
        )
