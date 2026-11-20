# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

"""Experimental full-turn feedback execution; no implicit tracker modification.

Numerical callback histories and memory live on the source context. The current
integrated output path deliberately uses ordinary Xdeps knob propagation and
small host command transfers. Readiness/finite predicates synchronise small
control flags. This is not a device-only or general event-time implementation.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import asdict, dataclass
import inspect
from itertools import chain
import numbers
from types import MappingProxyType
from weakref import ref

import numpy as np
import xobjects as xo

from .timing import (
    CommandQueue, DeadlineSchedule, FirstOrderActuator, _CalculationTiming,
    _nonnegative_float, _reached,
)
from .acquisition import Acquisition, _callback_configuration, _validity


_ARGUMENTS = {'inputs', 'parameters', 'previous', 'constants', 'event', 'xp'}
_ACTUATOR_READBACKS = {
    'applied_output', 'held_target', 'pending_output', 'total_target',
    'nominal_target',
}


def _callback_arguments(program):
    if not callable(program):
        raise TypeError('Feedback program must be callable')
    accepted = []
    for name, parameter in inspect.signature(program).parameters.items():
        if parameter.kind in (
            parameter.POSITIONAL_ONLY,
            parameter.VAR_POSITIONAL,
            parameter.VAR_KEYWORD,
        ):
            raise TypeError('Feedback programs use declared keyword arguments')
        if name not in _ARGUMENTS:
            if parameter.default is parameter.empty:
                raise TypeError(f'Unknown required feedback argument {name!r}')
        else:
            accepted.append(name)
    return tuple(accepted)


def _copy_mapping(mapping, xp, *, readonly=False):
    if not isinstance(mapping, Mapping):
        raise TypeError('Numerical state must be a mapping')
    result = {}
    for key, value in mapping.items():
        if not isinstance(key, str):
            raise TypeError('Feedback state keys must be strings')
        array = xp.asarray(value)
        if array.dtype.kind not in 'biufc':
            raise TypeError(f'Feedback value {key!r} is not numerical')
        array = array.copy()
        if readonly and isinstance(array, np.ndarray):
            array.flags.writeable = False
        result[key] = array
    return result


def _configuration(mapping, xp):
    result = {}
    for key, value in (mapping or {}).items():
        if not isinstance(key, str):
            raise TypeError('Configuration keys must be strings')
        # Host scalar metadata, including forecast loop bounds, stay host scalars.
        if isinstance(value, (numbers.Number, np.bool_)):
            if not np.isfinite(value):
                raise ValueError(f'Nonfinite configuration {key!r}')
            result[key] = value
        else:
            result.update(_copy_mapping({key: value}, xp, readonly=True))
            if not bool(xp.all(xp.isfinite(result[key]))):
                raise ValueError(f'Nonfinite configuration {key!r}')
    return MappingProxyType(result)


@dataclass
class _Candidate:
    controller: object
    history: dict
    source_schema: dict
    acquisition: object
    memory: dict
    queue: object
    actuator: object
    applied: np.ndarray
    schedule_event: object
    processor_free_time: float
    log_entry: dict


class FeedbackController:
    """Run a numerical program on completed records within one feedback system.

    Parameters
    ----------
    name : str
        Controller name, unique within its system.
    sources : mapping, optional
        Named sources on one tracking context. Each implements
        ``read_record(turn, *, completed_turn)`` and returns a record with
        ``value`` and ``valid``. At completed turn ``n``, the record turn
        is ``n - 1``. Sources may also come from ``acquisition``.
    program : callable
        Numerical calculation with named arguments selected from ``inputs``,
        ``parameters``, ``constants``, ``previous``, ``event`` and ``xp``.
        Return ``outputs`` with one command named by ``output_key``. Optional
        ``memory`` replaces the retained state and ``diagnostics`` enters
        the log. The calculation runs on the tracking context's array
        backend ``xp`` and may be reevaluated at a retried boundary.
    output : KnobOutput or PhysicalActuator
        Transactional binding from a scalar or vector command to Xdeps
        knobs. For a physical actuator the command is a contribution to
        its nominal setting.
    parameters, constants : mapping, optional
        Separate callback namespaces with the same snapshot semantics.
        Both are fixed settings for the run. Numeric scalars stay on the
        host and numerical arrays are owned on the tracking context.
    memory : mapping, optional
        Initial retained state. Its keys, shapes and dtypes stay fixed.
    history_length : int, optional
        Number of fresh publications needed before a calculation. Histories
        are exposed as ``<name>_history``, oldest first, when greater than one.
    update_turns : int, optional
        Number of reference turns between nominal calculation deadlines.
        The default is one when neither cadence is specified.
    update_frequency_hz : float, optional
        Nominal calculation rate in Hz. Use one cadence.
    first_calculation_turn, first_calculation_time_seconds : float, optional
        Absolute first deadline on the corresponding reference-turn or
        tracking-time clock. By default the grid starts one nominal interval
        after the run origin. Execute at the first ready boundary at or after
        a deadline and coalesce earlier missed deadlines into that update.
    processing_delay_seconds, command_delay_seconds : float, optional
        Serial calculation time and subsequent transport time. Delivery
        occurs at the first boundary at or after their sum. Processing
        overlaps raise an error.
    response_time_constant_seconds : float or array, optional
        Bare-output response time constant in seconds, scalar or per channel.
        Zero gives an immediate response in the absence of slew limits.
    maximum_slew_per_second : float or array, optional
        Bare-output slew limit in command units per second. A physical
        actuator owns the response and slew settings itself.
    slew_mode : {'coordinated', 'independent'}, optional
        Bare-output slew policy. The default scales the entire increment
        to meet the most restrictive channel limit.
    output_key : str, optional
        Name of the command entry in the program's ``outputs`` result.
    initial_time_seconds : float, optional
        Initial actuator time on the direct system clock. Set this to the
        run's starting timestamp when using direct boundaries at a nonzero
        epoch. Placed tracking supplies its own clock.
    log_capacity : int, optional
        Maximum number of retained boundary log entries.
    acquisition : Acquisition, optional
        Source bindings, selection cadence and optional window/transform.
        Additional sources can be supplied under distinct names.

    Notes
    -----
    Missing data on a selected passage clears history. Queued delivery
    and hardware response continue. Actuator response, limits and slew
    constrain fields. Stability and memory behavior under saturation
    depend on the controller law and the sampled beam response.
    """

    def __init__(
        self,
        name,
        *,
        sources=None,
        program,
        output,
        parameters=None,
        constants=None,
        memory=None,
        history_length=1,
        update_turns=None,
        update_frequency_hz=None,
        first_calculation_turn=None,
        first_calculation_time_seconds=None,
        processing_delay_seconds=0.0,
        command_delay_seconds=0.0,
        response_time_constant_seconds=0.0,
        maximum_slew_per_second=None,
        slew_mode=None,
        output_key='command',
        initial_time_seconds=0.0,
        log_capacity=1024,
        acquisition=None,
    ):
        if not isinstance(name, str) or not name:
            raise ValueError('Feedback controller needs a nonempty name')
        if not isinstance(output_key, str) or not output_key:
            raise ValueError('Feedback output_key must be a nonempty string')
        if acquisition is not None and not isinstance(acquisition, Acquisition):
            raise TypeError('acquisition must be an Acquisition configuration')
        bound_sources = {} if acquisition is None else dict(acquisition.sources)
        explicit_sources = {} if sources is None else dict(sources)
        if bound_sources.keys() & explicit_sources.keys():
            raise ValueError('Acquisition and controller source names overlap')
        sources = {**bound_sources, **explicit_sources}
        if not sources:
            raise ValueError('A feedback controller needs measurement sources')
        if (
            not isinstance(history_length, numbers.Integral)
            or isinstance(history_length, bool)
            or history_length < 1
        ):
            raise ValueError('history_length must be a positive host integer')
        if (
            not isinstance(log_capacity, numbers.Integral)
            or isinstance(log_capacity, bool)
            or log_capacity < 1
        ):
            raise ValueError('log_capacity must be a positive host integer')
        self.name = name
        self.sources = dict(sources)
        if any(not isinstance(key, str) or not key for key in self.sources):
            raise ValueError('Source names must be nonempty strings')
        self.context = getattr(next(iter(self.sources.values())), 'context', None)
        if self.context is None:
            raise TypeError('Feedback sources must declare a tracking context')
        if not isinstance(self.context, (xo.ContextCpu, xo.ContextCupy)):
            raise NotImplementedError('The prototype supports CPU and CuPy contexts')
        if any(
            getattr(source, 'context', None) is not self.context
            for source in self.sources.values()
        ):
            raise ValueError(
                'One controller currently requires a single source context'
            )
        for source in self.sources.values():
            if not callable(getattr(source, 'read_record', None)):
                raise TypeError('Feedback sources must provide read_record')
            stride = getattr(source, 'record_stride', 1)
            if (
                not isinstance(stride, numbers.Integral)
                or isinstance(stride, bool) or stride < 1
            ):
                raise ValueError('Source record stride must be a positive integer')
            if stride == 1:
                continue
            start = getattr(source, 'first_record_turn', None)
            if (
                acquisition is None or start is None
                or acquisition.sample_turns % stride
                or (acquisition.first_sample_turn - 1 - start) % stride
            ):
                raise ValueError('Acquisition cadence must align with source records')
        self.xp = self.context.nplike_lib
        self.program = program
        self._arguments = _callback_arguments(program)
        self.output = output
        self._system = None
        for method in ('validate', 'snapshot', 'apply',
                       '_checkpoint', '_restore_checkpoint'):
            if not callable(getattr(output, method, None)):
                raise TypeError(
                    'Integrated feedback requires a transactional knob output'
                )
        self.parameters = _configuration(parameters, self.xp)
        self.constants = _configuration(constants, self.xp)
        self.memory = _copy_mapping(memory or {}, self.xp)
        for key, value in self.memory.items():
            if not bool(self.xp.all(self.xp.isfinite(value))):
                raise ValueError(f'Initial feedback memory {key!r} must be finite')
        self._memory_schema = {
            key: (value.shape, value.dtype) for key, value in self.memory.items()
        }
        self.history_length = int(history_length)
        if any(name.endswith('_history') for name in self.sources):
            raise ValueError('Source names ending in _history are reserved')
        if _ACTUATOR_READBACKS & self.sources.keys():
            raise ValueError('Source name conflicts with an actuator readback')
        self.acquisition = None if acquisition is None else acquisition.copy()
        self._history = (
            {name: deque(maxlen=self.history_length) for name in self.sources}
            if acquisition is None
            else {}
        )
        self._source_schema = {}
        if update_turns is not None and update_frequency_hz is not None:
            raise ValueError('Specify exactly one feedback update cadence')
        if update_turns is None and update_frequency_hz is None:
            update_turns = 1
        if update_frequency_hz is None:
            if first_calculation_time_seconds is not None:
                raise ValueError('Use first_calculation_turn with a turn cadence')
            self.schedule = DeadlineSchedule(
                update_turns=update_turns, first_turn=first_calculation_turn,
            )
        else:
            if first_calculation_turn is not None:
                raise ValueError(
                    'Use first_calculation_time_seconds with a frequency cadence'
                )
            self.schedule = DeadlineSchedule(
                update_frequency_hz=update_frequency_hz,
                first_time=first_calculation_time_seconds,
            )
        self.processing_delay_seconds = _nonnegative_float(
            processing_delay_seconds, 'processing delay'
        )
        self.command_delay_seconds = _nonnegative_float(
            command_delay_seconds, 'command delay'
        )
        self._processor_free_time = _nonnegative_float(
            initial_time_seconds, 'initial time')
        self._last_calculation_time = None
        self.output_key = output_key
        self.queue = CommandQueue()
        # The physical output is on the host even when the response is on a GPU.
        self._applied = output.snapshot()
        if callable(getattr(output, 'make_actuator', None)):
            if (
                np.any(np.asarray(response_time_constant_seconds) != 0)
                or maximum_slew_per_second is not None
                or slew_mode is not None
            ):
                raise ValueError(
                    'Configure response and slew on the physical actuator owner'
                )
            self.actuator = output.make_actuator(
                xp=self.xp, initial_time_seconds=initial_time_seconds
            )
        else:
            self.actuator = FirstOrderActuator(
                self._applied,
                xp=self.xp,
                response_time_constant_seconds=response_time_constant_seconds,
                maximum_slew_per_second=maximum_slew_per_second,
                slew_mode='coordinated' if slew_mode is None else slew_mode,
                initial_time_seconds=initial_time_seconds,
            )
        self.enabled = True
        self._requires_reconciliation = False
        self.log = deque(maxlen=int(log_capacity))

    def disable(self, *, cancel_pending=False):
        """Pause calculations. Pending delivery and field response continue."""
        self._check_stopped()
        self.enabled = False
        if cancel_pending:
            self._requires_reconciliation |= bool(self.memory) and bool(len(self.queue))
            self.queue.clear()

    def enable(self, *, memory=None):
        """Resume; explicitly reconcile retained state after cancelling requests.

        Elapsed callback dt includes the disabled interval. This method does
        not promise bumpless restart or select a generic anti-windup law.
        """
        self._check_stopped()
        if self._requires_reconciliation and memory is None:
            raise RuntimeError(
                'Cancelled requests require explicit controller memory reconciliation'
            )
        if memory is not None:
            memory = _copy_mapping(memory, self.xp)
            schema = {key: (value.shape, value.dtype) for key, value in memory.items()}
            if schema != self._memory_schema or any(
                not bool(self.xp.all(self.xp.isfinite(value)))
                for value in memory.values()
            ):
                raise ValueError('Reconciled memory must preserve finite state schema')
            self.memory = memory
        self._requires_reconciliation = False
        self.enabled = True

    def _check_stopped(self):
        system = self._system() if self._system is not None else None
        if system is not None and system._busy:
            raise RuntimeError('Change controller activation between boundaries')

    def _read_sources(self, completed_turn):
        """Read one certified passage per source on the tracking context."""
        xp = self.xp
        raw = {}
        record_ids = {}
        all_valid = True
        for name, source in self.sources.items():
            record = source.read_record(
                completed_turn - 1, completed_turn=completed_turn
            )
            value = xp.asarray(record.value)
            # Require all selected bins or particles to be present.
            all_valid = all_valid & _validity(record.valid, xp)
            raw[name] = value
            record_ids[name] = getattr(record, 'record_index', completed_turn - 1)
        return raw, record_ids, all_valid

    def _calculate(self, inputs, event_values, completed_turn):
        """Evaluate a callback on snapshots and validate its result."""
        xp = self.xp
        try:
            values = dict(inputs=MappingProxyType(inputs),
                          event=MappingProxyType(event_values), xp=xp)
            if 'previous' in self._arguments:
                values['previous'] = MappingProxyType(
                    _copy_mapping(self.memory, xp, readonly=True))
            for name in ('parameters', 'constants'):
                if name in self._arguments:
                    values[name] = _callback_configuration(getattr(self, name), xp)
            result = self.program(**{name: values[name] for name in self._arguments})
            if not isinstance(result, Mapping):
                raise TypeError('Feedback program must return a result mapping')
            if set(result) - {'memory', 'outputs', 'diagnostics'}:
                raise ValueError('Unknown feedback result mapping keys')
            outputs = result.get('outputs')
            if not isinstance(outputs, Mapping) or set(outputs) != {
                self.output_key
            }:
                raise ValueError(
                    'Declare exactly the configured feedback output key'
                )
            memory = _copy_mapping(result.get('memory', self.memory), xp)
            schema = {
                key: (value.shape, value.dtype) for key, value in memory.items()
            }
            if schema != self._memory_schema:
                raise ValueError('Feedback memory keys, shapes or dtypes changed')
            if any(
                not bool(xp.all(xp.isfinite(value))) for value in memory.values()
            ):
                raise ValueError('Feedback memory must be finite')
            requested = self.output.validate(outputs[self.output_key])
            diagnostics = result.get('diagnostics', {})
            if not isinstance(diagnostics, Mapping):
                raise TypeError('Feedback diagnostics must be a mapping')
            diagnostics = _copy_mapping(diagnostics, xp)
        except Exception as error:
            error.add_note(
                f'Feedback controller {self.name!r} at completed turn {completed_turn}'
            )
            raise
        return memory, requested, diagnostics

    def _preview(
        self, completed_turn, time_seconds, model_time_seconds, clock_origin,
    ):
        """Stage completed measurements, one calculation and hardware response."""
        xp = self.xp
        history = {
            name: deque(values, maxlen=self.history_length)
            for name, values in self._history.items()
        }
        if self.acquisition is None or self.acquisition._samples_turn(completed_turn):
            raw, record_ids, valid = self._read_sources(completed_turn)
        else:
            raw, record_ids, valid = None, None, True
        source_schema = self._source_schema
        if raw is not None:
            source_schema = {
                name: (value.shape, value.dtype.str) for name, value in raw.items()
            }
        if self._source_schema and source_schema != self._source_schema:
            raise ValueError('Feedback source names, shapes or dtypes changed')
        acquisition = None
        if self.acquisition is None:
            for value in raw.values():
                valid = valid & xp.all(xp.isfinite(value))
            valid = bool(valid)
            published = valid
            invalidated = not valid
            inputs = {name: value.copy() for name, value in raw.items()}
        else:
            acquisition = self.acquisition.preview(
                raw,
                completed_turn=completed_turn,
                time_seconds=time_seconds,
                valid=valid,
                parameters=self.parameters,
                constants=self.constants,
                record_ids=record_ids,
                xp=xp,
            )
            inputs = dict(acquisition.samples)
            valid = acquisition.valid
            published = acquisition.published
            invalidated = acquisition.invalidated
            if any(
                name.endswith('_history') or name in _ACTUATOR_READBACKS
                for name in inputs
            ):
                raise ValueError('Acquisition sample name is reserved by the runtime')
        if invalidated:
            for values in history.values():
                values.clear()
        if published:
            if not history:
                history = {name: deque(maxlen=self.history_length) for name in inputs}
            if set(history) != set(inputs):
                raise ValueError('Acquisition sample names changed')
            for name, value in inputs.items():
                history[name].append(value.copy())
        ready = (
            valid
            and bool(history)
            and all(len(values) == self.history_length for values in history.values())
        )
        # Default grids start at the run origin. Explicit phases stay absolute.
        absolute_phase = self.schedule._explicit_first_deadline
        schedule_time = time_seconds if absolute_phase else model_time_seconds
        event = self.schedule.peek(
            time_seconds=schedule_time, reference_turns=completed_turn,
        )
        execute = self.enabled and ready and event is not None
        queue = self.queue.copy()
        memory = self.memory
        processor_free_time = self._processor_free_time
        diagnostics = {}
        requested = None
        if execute:
            if self.history_length > 1:
                inputs.update(
                    {
                        name + '_history': xp.stack(tuple(values), axis=0)
                        for name, values in history.items()
                    }
                )
            if not _reached(model_time_seconds, self._processor_free_time):
                raise ValueError(
                    f'Feedback {self.name!r} processing interval overlaps '
                    'its next calculation'
                )
            pending = self.queue.pending
            total_target = self.actuator.target
            contribution = getattr(self.actuator, 'contribution_target', None)
            held_target = total_target if contribution is None else contribution
            inputs['applied_output'] = self.actuator.applied
            inputs['held_target'] = held_target
            if contribution is not None:
                inputs['total_target'] = total_target
                inputs['nominal_target'] = self.actuator.nominal_target
            inputs['pending_output'] = (
                xp.asarray(pending[-1].value).copy() if pending else held_target.copy()
            )
            dt_seconds = (
                0.0 if self._last_calculation_time is None
                else model_time_seconds - self._last_calculation_time
            )
            nominal_time = event.nominal_time_seconds
            if nominal_time is not None and not absolute_phase:
                nominal_time += clock_origin
            event_values = asdict(event)
            event_values['nominal_time_seconds'] = nominal_time
            event_values.update(_CalculationTiming(
                completed_turn=completed_turn,
                time_seconds=time_seconds,
                dt_seconds=dt_seconds,
            ))
            event_values['pending_count'] = len(pending)
            event_values['pending_ready_times_seconds'] = tuple(
                clock_origin + command.ready_time_seconds for command in pending
            )
            if acquisition is not None:
                event_values.update(
                    acquisition_first_completed_turn=acquisition.first_completed_turn,
                    acquisition_last_completed_turn=acquisition.last_completed_turn,
                    acquisition_start_time_seconds=acquisition.start_time_seconds,
                    acquisition_end_time_seconds=acquisition.end_time_seconds,
                    acquisition_sample_count=acquisition.sample_count,
                    acquisition_published=acquisition.published,
                    acquisition_record_ids=acquisition.record_ids,
                )
            memory, requested, diagnostics = self._calculate(
                inputs, event_values, completed_turn,
            )
            queue.enqueue(
                requested,
                time_seconds=model_time_seconds,
                processing_delay_seconds=self.processing_delay_seconds,
                command_delay_seconds=self.command_delay_seconds,
            )
            processor_free_time = model_time_seconds + self.processing_delay_seconds
        commands = queue.peek_ready(model_time_seconds)
        target = commands[-1].value if commands else None
        actuator = self.actuator.preview(model_time_seconds, target=target)
        # Also validates the applied value when only hardware evolution is due.
        validate_applied = getattr(
            self.output, 'validate_applied', self.output.validate
        )
        applied = validate_applied(actuator.applied)
        queue.commit_ready(commands, time_seconds=model_time_seconds)
        entry = dict(
            completed_turn=completed_turn,
            time_seconds=time_seconds,
            executed=execute,
            source_valid=valid,
            acquisition_published=published,
            requested_output=None if requested is None else requested.copy(),
            applied_output=actuator.applied.copy(),
            target_output=actuator.target.copy(),
            pending_commands=len(queue),
            diagnostics=diagnostics,
        )
        if execute:
            entry.update(
                sequence_index=event.sequence_index,
                dt_seconds=dt_seconds,
                nominal_time_seconds=nominal_time,
                nominal_reference_turns=event.nominal_reference_turns,
                missed_deadlines=event.missed_deadlines,
            )
        return _Candidate(
            controller=self,
            history=history,
            source_schema=source_schema,
            acquisition=acquisition,
            memory=memory,
            queue=queue,
            actuator=actuator,
            applied=applied,
            schedule_event=event if execute else None,
            processor_free_time=processor_free_time,
            log_entry=entry,
        )


class FeedbackSystem:
    """Commit controllers together after one completed traversal.

    Each controller belongs to one system. A failed calculation or a
    successful field rollback permits retry at the same stopped boundary.
    An internal commit or rollback failure requires a fresh system and
    coherent state. Callbacks and Xdeps expressions must be pure and
    outputs must be owned exclusively at the boundary.
    """

    def __init__(self, controllers, *, lines=None, _turn_clock=None):
        self._controllers = tuple(controllers)
        self._turn_clock = _turn_clock
        if any(not isinstance(c, FeedbackController) for c in self.controllers):
            raise TypeError('Bind FeedbackController instances')
        if not self.controllers or len({c.name for c in self.controllers}) != len(
            self.controllers
        ):
            raise ValueError('Feedback controllers must have unique names')
        if any(c._system is not None for c in self.controllers):
            raise ValueError('A controller can belong to only one feedback system')
        self._check_ownership()
        self._source_bindings = []
        self._controller_bindings = [
            (controller, controller.output, dict(controller.sources))
            for controller in self.controllers
        ]
        self.lines = MappingProxyType(
            {}
            if lines is None
            else dict(lines)
            if isinstance(lines, Mapping)
            else {str(index): line for index, line in enumerate(lines)}
        )
        for controller in self.controllers:
            for source in controller.sources.values():
                monitor = getattr(source, 'monitor', None)
                if monitor is None:
                    continue  # Custom sources certify their own records.
                if not self.lines:
                    raise ValueError(
                        'Bind the participating tracking lines for monitor sources'
                    )
                matches = [
                    (line, name)
                    for line in self.lines.values()
                    for name in line.element_names
                    if self._same_element(line.get(name), monitor)
                ]
                if len(matches) != 1:
                    raise ValueError(
                        'Each monitor source must occur once in participating lines'
                    )
                line, name = matches[0]
                if line.tracker is None:
                    raise ValueError('Build source lines before binding feedback')
                self._source_bindings.append((source, line, name, line.tracker))
        self._completed_turn = None
        self._time_seconds = None
        self._busy = False
        self._failure = None
        for controller in self.controllers:
            controller._system = ref(self)

    @property
    def controllers(self):
        """Controllers bound to this transaction."""
        return self._controllers

    @staticmethod
    def _same_element(first, second):
        return first is second or (
            hasattr(first, '_buffer')
            and hasattr(second, '_buffer')
            and first._buffer is second._buffer
            and first._offset == second._offset
            and type(first) is type(second)
        )

    def _check_sources(self):
        for controller, output, sources in self._controller_bindings:
            if controller.output is not output:
                raise RuntimeError('Output binding changed; reconstruct feedback')
            if controller.sources.keys() != sources.keys() or any(
                controller.sources[name] is not source
                for name, source in sources.items()
            ):
                raise RuntimeError(
                    'Source bindings changed; reconstruct the feedback system'
                )
        for source, line, name, tracker in self._source_bindings:
            if line.tracker is not tracker or not self._same_element(
                line.get(name), source.monitor
            ):
                raise RuntimeError(
                    'Tracking/source binding changed; reconstruct the feedback system'
                )

    def _check_ownership(self, other_outputs=()):
        seen = set()
        setpoints = set()
        outputs = (controller.output for controller in self.controllers)
        for output in chain(outputs, other_outputs):
            keys = set(output.ownership_keys)
            keys.update(getattr(output, 'dependent_field_keys', ()))
            external = set(getattr(output, 'external_setpoint_keys', ()))
            if setpoints & keys or external & seen or external & keys:
                raise ValueError(
                    'Nominal setpoints require external ownership and cannot '
                    'write controlled fields'
                )
            if seen & keys:
                raise ValueError(
                    'Feedback outputs share a contribution or physical field'
                )
            seen.update(keys)
            setpoints.update(external)

    def _check_applied_outputs(self):
        """Reject stopped edits to applied knobs before a new traversal."""
        for controller in self.controllers:
            installed = controller.output.snapshot()
            if not np.array_equal(installed, controller._applied):
                raise RuntimeError(
                    'Applied actuator knob changed outside its response owner')

    def step(self, completed_turn, time_seconds):
        """Commit a completed traversal at its absolute tracking time in seconds.

        ``completed_turn`` is a consecutive reference-turn count. The first
        direct boundary follows the run origin by one full traversal.
        """
        if self._failure is not None:
            raise RuntimeError(
                'Feedback system is unusable after an internal commit/rollback failure'
            ) from self._failure
        if self._busy:
            raise RuntimeError('A feedback boundary transaction cannot reenter')
        if (
            not isinstance(completed_turn, numbers.Integral)
            or isinstance(completed_turn, bool)
            or completed_turn < 1
        ):
            raise ValueError('completed_turn must be a positive host integer')
        time_seconds = _nonnegative_float(time_seconds, 'boundary time')
        if self._completed_turn is not None:
            if (
                completed_turn != self._completed_turn + 1
                or time_seconds <= self._time_seconds
            ):
                raise ValueError(
                    'Call feedback once per increasing completed full turn'
                )
        self._busy = True
        try:
            self._check_sources()
            self._check_ownership()
            self._check_applied_outputs()
            first_turn = completed_turn - 1
            model_time_seconds = time_seconds
            clock_origin = 0.0
            if self._turn_clock is not None:
                # Compare relative intervals without the large absolute epoch.
                first_turn, clock_origin, period = self._turn_clock
                model_time_seconds = (completed_turn - first_turn) * period
            if self._completed_turn is None:
                for controller in self.controllers:
                    controller.schedule._start_at(
                        reference_turns=first_turn,
                        time_seconds=controller.actuator.time_seconds,
                    )
            candidates = [
                c._preview(
                    completed_turn, time_seconds, model_time_seconds, clock_origin,
                )
                for c in self.controllers
            ]
            self._check_sources()
            writes = []
            for candidate in candidates:
                controller = candidate.controller
                output = controller.output
                if not np.array_equal(candidate.applied, controller._applied):
                    writes.append((output, output._checkpoint(), candidate.applied))
            written = []
            committing = False
            try:
                for output, old, applied in writes:
                    written.append((output, old))
                    output.apply(applied)
                committing = True
                for candidate in candidates:
                    controller = candidate.controller
                    controller._history = candidate.history
                    controller._source_schema = candidate.source_schema
                    if candidate.acquisition is not None:
                        controller.acquisition.commit(candidate.acquisition)
                    controller.memory = candidate.memory
                    controller.queue = candidate.queue
                    controller.actuator.commit(candidate.actuator)
                    controller._applied = candidate.applied
                    if candidate.schedule_event is not None:
                        controller.schedule.commit(candidate.schedule_event)
                        controller._last_calculation_time = model_time_seconds
                    controller._processor_free_time = candidate.processor_free_time
                    controller.log.append(candidate.log_entry)
            except BaseException as original:
                failures = []
                for output, old in reversed(written):
                    try:
                        output._restore_checkpoint(old)
                    except BaseException as error:
                        failures.append(error)
                if failures:
                    self._failure = original
                    error = RuntimeError(
                        'Feedback output rollback failed; tracking must remain stopped'
                    )
                    for failure in failures:
                        error.add_note(str(failure))
                    raise error from original
                if committing:
                    self._failure = original
                    raise RuntimeError(
                        'Internal feedback commit failed; system cannot be retried'
                    ) from original
                raise
            self._completed_turn = int(completed_turn)
            self._time_seconds = time_seconds
            return tuple(candidate.log_entry for candidate in candidates)
        finally:
            self._busy = False
