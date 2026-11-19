# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

"""Feedback deadlines and command transport."""

from dataclasses import dataclass, replace
import heapq
import math
import numbers
import operator

import numpy as np


# Relative allowance for roundoff in independently advanced branch clocks.
# Admission caps it at 1% of a revolution. Deadlines use two ULPs.
_CLOCK_ROUNDOFF = 128 * np.finfo(float).eps


def _nonnegative_float(value, name):
    """Validate and convert a host clock value, interval or frequency."""
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative')
    return value


def _reached(now, deadline):
    """Compare seconds-based timestamps with a two-ULP allowance."""
    # Reconstruct fixed-period boundaries to avoid accumulated clock drift.
    return (
        math.isfinite(deadline)
        and deadline - now <= 2 * math.ulp(max(now, deadline))
    )


def _check_turn_clock_resolution(time_seconds, revolution_period):
    """Reject an epoch too large for the shared full-turn clock."""
    time_seconds = _nonnegative_float(time_seconds, 'Feedback boundary time')
    if _CLOCK_ROUNDOFF * time_seconds > revolution_period / 100:
        raise ValueError(
            'Feedback epoch has insufficient floating-point clock resolution')


@dataclass(frozen=True)
class ScheduleEvent:
    """One update opportunity, with the first due deadline and last due index."""

    time_seconds: float
    reference_turns: float
    dt_seconds: float
    sequence_index: int
    deadline_index: int
    nominal_time_seconds: float | None
    nominal_reference_turns: float | None
    missed_deadlines: int


class DeadlineSchedule:
    """Schedule updates on a fixed grid in seconds or reference turns.

    The first deadline defaults to one update interval from the clock origin.
    """

    def __init__(
        self,
        *,
        update_turns=None,
        update_frequency_hz=None,
        first_time=None,
        first_turn=None,
        missed_policy='coalesce',
    ):
        if (update_turns is None) == (update_frequency_hz is None):
            raise ValueError('Specify exactly one update cadence')
        if missed_policy not in ('coalesce', 'raise'):
            raise ValueError('Unknown missed-deadline policy')
        self._turn_based = update_turns is not None
        if self._turn_based:
            if (
                not isinstance(update_turns, numbers.Integral)
                or isinstance(update_turns, bool)
                or update_turns <= 0
            ):
                raise ValueError('update_turns must be a positive host integer')
            if first_time is not None:
                raise ValueError('Use first_turn with a turn schedule')
            self._period = float(update_turns)
            self._origin = (self._period if first_turn is None
                            else _nonnegative_float(first_turn, 'first_turn'))
        else:
            frequency = _nonnegative_float(update_frequency_hz,
                                           'update_frequency_hz')
            if frequency == 0:
                raise ValueError('update_frequency_hz must be positive')
            if first_turn is not None:
                raise ValueError('Use first_time with a frequency schedule')
            self._period = 1.0 / frequency
            if not math.isfinite(self._period):
                raise ValueError('Feedback update interval must be finite')
            self._origin = (self._period if first_time is None
                            else _nonnegative_float(first_time, 'first_time'))
        self.missed_policy = missed_policy
        self._next_index = 0
        self._sequence = 0
        self._last_time = None
        self._last_turn = None

    @property
    def next_deadline(self):
        """Next nominal deadline in seconds or reference turns, as configured."""
        return self._origin + self._next_index * self._period

    def peek(self, *, time_seconds, reference_turns):
        """Preview a due update without advancing the schedule.

        Coalesce missed deadlines into the update or raise, as configured.
        """
        time_seconds = _nonnegative_float(time_seconds, 'time_seconds')
        reference_turns = _nonnegative_float(reference_turns, 'reference_turns')
        if self._last_time is not None and (
            time_seconds < self._last_time or reference_turns < self._last_turn
        ):
            raise ValueError('Feedback schedule cannot move backwards')
        if self._last_time == time_seconds and self._last_turn == reference_turns:
            return None
        coordinate = reference_turns if self._turn_based else time_seconds
        reached = operator.ge if self._turn_based else _reached
        if not reached(coordinate, self.next_deadline):
            return None
        # Keep the roundoff margins of neighboring deadlines separate.
        if self._period <= 4 * math.ulp(max(coordinate, self._origin)):
            raise ValueError('Feedback update interval is below clock resolution')
        # Use nominal deadlines to preserve phase when a boundary is late.
        index = max(
            self._next_index,
            int(math.floor((coordinate - self._origin) / self._period)),
        )
        # Division can round across an integer. Recheck the adjacent grid point.
        if reached(coordinate, self._origin + (index + 1) * self._period):
            index += 1
        elif not reached(coordinate, self._origin + index * self._period):
            index -= 1
        missed = index - self._next_index
        if missed and self.missed_policy == 'raise':
            raise ValueError(f'Execution opportunity missed {missed} nominal deadlines')
        nominal = self._origin + self._next_index * self._period
        return ScheduleEvent(
            time_seconds=time_seconds,
            reference_turns=reference_turns,
            dt_seconds=0.0
            if self._last_time is None
            else time_seconds - self._last_time,
            sequence_index=self._sequence,
            deadline_index=index,
            nominal_time_seconds=None if self._turn_based else nominal,
            nominal_reference_turns=nominal if self._turn_based else None,
            missed_deadlines=missed,
        )

    def commit(self, event):
        """Consume all deadlines covered by the previewed event."""
        expected = self.peek(
            time_seconds=event.time_seconds, reference_turns=event.reference_turns
        )
        if expected != event:
            raise ValueError('Stale or invalid schedule event')
        self._next_index = event.deadline_index + 1
        self._last_time = event.time_seconds
        self._last_turn = event.reference_turns
        self._sequence += 1

    def poll(self, *, time_seconds, reference_turns):
        """Return and consume the update due at this boundary, if any."""
        event = self.peek(time_seconds=time_seconds, reference_turns=reference_turns)
        if event is not None:
            self.commit(event)
        return event


@dataclass(frozen=True)
class QueuedCommand:
    """A command snapshot with processing and transport delays in seconds."""

    value: object
    requested_time_seconds: float
    ready_time_seconds: float
    sequence_index: int
    processing_delay_seconds: float
    command_delay_seconds: float


class CommandQueue:
    """Deliver commands at sampled boundaries in arrival order.

    Commands with the same arrival time are delivered in enqueue order.
    """

    def __init__(self):
        self._heap = []
        self._sequence = 0
        self._last_delivery_time = None

    def __len__(self):
        return len(self._heap)

    @property
    def pending(self):
        """Return copies of all pending commands in delivery order."""
        return tuple(
            replace(command, value=command.value.copy())
            for _, _, command in sorted(self._heap)
        )

    def copy(self):
        """Fork the pending queue for a trial update."""
        result = type(self)()
        # The queue only exposes copies of its owned command arrays.
        result._heap = self._heap.copy()
        result._sequence = self._sequence
        result._last_delivery_time = self._last_delivery_time
        return result

    def enqueue(
        self,
        value,
        *,
        time_seconds,
        processing_delay_seconds=0.0,
        command_delay_seconds=0.0,
    ):
        """Queue a copy of the value and return its sequence index."""
        requested = _nonnegative_float(time_seconds, 'time_seconds')
        processing = _nonnegative_float(
            processing_delay_seconds, 'processing_delay_seconds')
        transport = _nonnegative_float(command_delay_seconds,
                                       'command_delay_seconds')
        if (
            self._last_delivery_time is not None
            and requested < self._last_delivery_time
        ):
            raise ValueError('Cannot backdate a newly calculated command')
        # Sum the delays before adding a potentially large clock epoch.
        ready = requested + (processing + transport)
        if not math.isfinite(ready):
            raise ValueError('Command ready time overflowed')
        # Convert host sequences while retaining context-native arrays.
        if not (hasattr(value, 'shape') and hasattr(value, 'copy')):
            value = np.asarray(value)
        sequence = self._sequence
        command = QueuedCommand(
            value.copy(), requested, ready, sequence, processing, transport
        )
        heapq.heappush(self._heap, (ready, sequence, command))
        self._sequence += 1
        return sequence

    def _due(self, time_seconds):
        """Yield internal commands eligible at a sampled boundary."""
        boundary = _nonnegative_float(time_seconds, 'time_seconds')
        if self._last_delivery_time is not None and boundary < self._last_delivery_time:
            raise ValueError('Command delivery cannot move backwards')
        for ready, _, command in sorted(self._heap):
            if not _reached(boundary, ready):
                break
            positive_delay = (
                command.processing_delay_seconds > 0
                or command.command_delay_seconds > 0
            )
            # A positive delay may round to the request's timestamp.
            request = command.requested_time_seconds
            if boundary > request or (boundary == request and not positive_delay):
                yield command

    def peek_ready(self, time_seconds):
        """Return copies of due commands without consuming them."""
        return tuple(
            replace(command, value=command.value.copy())
            for command in self._due(time_seconds)
        )

    def commit_ready(self, commands, *, time_seconds):
        """Consume the preview if it still matches the due commands."""
        ids = tuple(command.sequence_index for command in commands)
        if ids != tuple(command.sequence_index for command in self._due(time_seconds)):
            raise ValueError('Stale or incomplete ready-command transaction')
        selected = set(ids)
        self._heap = [entry for entry in self._heap if entry[1] not in selected]
        heapq.heapify(self._heap)
        self._last_delivery_time = float(time_seconds)

    def ready(self, time_seconds):
        """Deliver and remove the commands due at this boundary."""
        commands = self.peek_ready(time_seconds)
        self.commit_ready(commands, time_seconds=time_seconds)
        return commands

    def clear(self):
        """Cancel pending commands while preserving clock and sequence identity."""
        self._heap.clear()
