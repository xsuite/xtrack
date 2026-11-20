# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

"""Reduce selected completed passages on the tracking context."""

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
import inspect
import math
import numbers
from types import MappingProxyType

import numpy as np

from .readout import bind_monitor


_TRANSFORM_ARGUMENTS = {'inputs', 'parameters', 'constants', 'event', 'xp'}


def _positive_integer(value, name):
    """Validate a positive host index or count."""
    if (
        not isinstance(value, numbers.Integral)
        or isinstance(value, bool)
        or value < 1
    ):
        raise ValueError(f'{name} must be a positive host integer')
    return int(value)


def _transform_arguments(transform):
    """Return the acquisition arguments requested by the callback."""
    if transform is None:
        return ()
    if not callable(transform):
        raise TypeError('Acquisition transform must be callable')
    arguments = []
    for name, parameter in inspect.signature(transform).parameters.items():
        if parameter.kind in (
            parameter.POSITIONAL_ONLY,
            parameter.VAR_POSITIONAL,
            parameter.VAR_KEYWORD,
        ):
            raise TypeError(
                'Acquisition transforms use declared keyword arguments')
        if name not in _TRANSFORM_ARGUMENTS:
            if parameter.default is parameter.empty:
                raise TypeError(f'Unknown acquisition argument {name!r}')
        else:
            arguments.append(name)
    return tuple(arguments)


def _arrays(mapping, xp, *, copy=True, readonly=False):
    """Validate context arrays and optionally take owned snapshots."""
    if not isinstance(mapping, Mapping) or not mapping:
        raise ValueError('Expected a nonempty mapping of acquisition arrays')
    result = {}
    for name, value in mapping.items():
        if not isinstance(name, str) or not name:
            raise ValueError('Acquisition names must be nonempty strings')
        array = xp.asarray(value)
        if array.dtype.kind not in 'biufc':
            raise TypeError(f'Acquisition value {name!r} is not numerical')
        if array.size == 0:
            raise ValueError(f'Acquisition value {name!r} has no channels')
        if copy:
            array = array.copy()
        if readonly and isinstance(array, np.ndarray):
            array.flags.writeable = False
        result[name] = array
    return result


def _schema(mapping):
    """Describe each named array without transferring its values."""
    return {
        name: (value.shape, value.dtype.str) for name, value in mapping.items()
    }


def _validity(value, xp):
    """Require a nonempty Boolean predicate and reduce it to one flag."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    predicate = xp.asarray(value)
    if predicate.dtype.kind != 'b' or predicate.size == 0:
        raise TypeError('Validity must be a nonempty Boolean predicate')
    return xp.all(predicate)


def _callback_configuration(mapping, xp):
    """Isolate mutable configuration while keeping host scalars scalar."""
    result = {}
    for name, value in (mapping or {}).items():
        if isinstance(value, (numbers.Number, np.bool_)):
            result[name] = value
        else:
            result.update(_arrays({name: value}, xp, readonly=True))
    return MappingProxyType(result)


@dataclass(frozen=True)
class _Passage:
    """Owned values and completion metadata for one selected passage."""

    turn: int
    time_seconds: float
    values: dict
    record_ids: dict


@dataclass
class _State:
    window: deque
    raw_schema: dict | None = None
    sample_schema: dict | None = None
    last_boundary_turn: int | None = None
    last_boundary_time: float | None = None
    last_due_turn: int | None = None
    highest_record_ids: dict | None = None
    latest_samples: dict = field(default_factory=dict)
    latest_valid: bool = False


@dataclass(frozen=True)
class AcquisitionCandidate:
    """A proposed update and the completion times of its selected passages.

    Append only new ``published`` samples. Reset consecutive controller history
    on ``invalidated``, even when the same update publishes a fresh sample.
    """

    samples: Mapping
    valid: bool
    published: bool
    invalidated: bool
    first_completed_turn: int | None
    last_completed_turn: int | None
    start_time_seconds: float | None
    end_time_seconds: float | None
    sample_count: int
    record_ids: Mapping
    _state: _State = field(repr=False, compare=False)
    _xp: object = field(repr=False, compare=False)
    _owner: object = field(repr=False, compare=False)
    _generation: int = field(repr=False, compare=False)


class Acquisition:
    """Rolling reduction of the last ``window_samples`` selected records.

    Select every ``sample_turns`` passage from ``first_sample_turn`` and
    publish on each valid selection once the window is full. The controller
    sets its own calculation cadence.

    Bind sources or ``(monitor, quantity)`` pairs after building the tracker.
    Bound sources use one tracking context. Each supplied record index must
    increase for its source. A missing, invalid or replayed selected passage
    clears the window and publication.

    The optional transform receives the reduced channels and read-only
    completion metadata. It returns a sample mapping and a validity flag.
    """

    def __init__(
        self,
        *,
        sources=None,
        window_samples=1,
        sample_turns=1,
        first_sample_turn=1,
        reduction='sum',
        transform=None,
    ):
        if sources is not None and not isinstance(sources, Mapping):
            raise TypeError('Acquisition sources must be a named mapping')
        bound = {}
        for name, source in (sources or {}).items():
            if not isinstance(name, str) or not name:
                raise ValueError('Source names must be nonempty strings')
            if isinstance(source, tuple) and len(source) == 2:
                source = bind_monitor(*source)
            if not hasattr(source, 'context') or not callable(
                getattr(source, 'read_record', None)
            ):
                raise TypeError('A source needs context and read_record')
            bound[name] = source
        self._sources = MappingProxyType(bound)
        self._source_context = next(
            (source.context for source in bound.values()), None)
        if any(
            source.context is not self._source_context
            for source in bound.values()
        ):
            raise ValueError('Acquisition sources must share one context')
        self.window_samples = _positive_integer(
            window_samples, 'window_samples')
        self.sample_turns = _positive_integer(sample_turns, 'sample_turns')
        self.first_sample_turn = _positive_integer(
            first_sample_turn, 'first_sample_turn'
        )
        if reduction not in ('sum', 'mean'):
            raise ValueError("Acquisition reduction must be 'sum' or 'mean'")
        self.reduction = reduction
        self.transform = transform
        self._arguments = _transform_arguments(transform)
        self._configuration = (
            self.window_samples,
            self.sample_turns,
            self.first_sample_turn,
            self.reduction,
            self.transform,
        )
        self._state = _State(window=deque(maxlen=self.window_samples))
        self._xp = None
        self._owner = object()
        self._generation = 0
        self._transform_active = False

    @property
    def sources(self):
        """Read-only completed-record bindings, shared by acquisition copies."""
        return self._sources

    @classmethod
    def from_monitor(cls, monitor, quantities, *, bunch_slot=None, **kwargs):
        """Bind named monitor quantities; reduce only as explicitly configured.

        For a weighted BPM window, bind ``{'weight': 'num_particles',
        'sum_y': 'sum_y'}`` and transform their reduced values into a ratio.
        """
        if not isinstance(quantities, Mapping) or not quantities:
            raise ValueError('Monitor quantities need a nonempty named mapping')
        sources = {
            name: bind_monitor(
                monitor, quantity, bunch_slot=bunch_slot)
            for name, quantity in quantities.items()
        }
        return cls(sources=sources, **kwargs)

    def copy(self):
        """Copy the acquisition with independent window and commit state."""
        self._check_configuration()
        result = type(self)(
            sources=self.sources,
            window_samples=self.window_samples,
            sample_turns=self.sample_turns,
            first_sample_turn=self.first_sample_turn,
            reduction=self.reduction,
            transform=self.transform,
        )
        result._state = replace(self._state)
        result._xp = self._xp
        return result

    def _check_configuration(self):
        if self._transform_active:
            raise RuntimeError('Acquisition cannot reenter its transform')
        current = (
            self.window_samples,
            self.sample_turns,
            self.first_sample_turn,
            self.reduction,
            self.transform,
        )
        if current != self._configuration:
            raise RuntimeError(
                'Acquisition configuration changed; bind a new acquisition'
            )

    @staticmethod
    def _clear(state):
        state.window = deque(maxlen=state.window.maxlen)
        state.latest_samples = {}
        state.latest_valid = False

    def _candidate(self, state, xp, published, invalidated):
        samples = (
            {name: value.copy() for name, value in state.latest_samples.items()}
            if state.latest_valid
            else {}
        )
        ids = {
            name: tuple(passage.record_ids[name] for passage in state.window)
            for name in state.raw_schema
        }
        first = state.window[0] if state.window else None
        last = state.window[-1] if state.window else None
        return AcquisitionCandidate(
            samples=MappingProxyType(samples),
            valid=state.latest_valid,
            published=published,
            invalidated=invalidated,
            first_completed_turn=first.turn if first is not None else None,
            last_completed_turn=last.turn if last is not None else None,
            start_time_seconds=first.time_seconds if first else None,
            end_time_seconds=last.time_seconds if last is not None else None,
            sample_count=len(state.window),
            record_ids=MappingProxyType(ids),
            _state=state,
            _xp=xp,
            _owner=self._owner,
            _generation=self._generation,
        )

    def _transform_sample(self, reduced, window, parameters, constants, xp):
        """Call the reduction transform with a snapshot of its arguments."""
        arguments = {'xp': xp}
        if 'inputs' in self._arguments:
            arguments['inputs'] = MappingProxyType(
                _arrays(reduced, xp, copy=False, readonly=True))
        if 'event' in self._arguments:
            window_ids = {
                name: tuple(passage.record_ids[name] for passage in window)
                for name in reduced
            }
            first, last = window[0], window[-1]
            arguments['event'] = MappingProxyType(dict(
                completed_turn=last.turn,
                time_seconds=last.time_seconds,
                first_completed_turn=first.turn,
                last_completed_turn=last.turn,
                start_time_seconds=first.time_seconds,
                end_time_seconds=last.time_seconds,
                sample_count=len(window),
                record_ids=MappingProxyType(window_ids),
            ))
        configuration = (
            ('parameters', parameters), ('constants', constants),
        )
        for name, mapping in configuration:
            if name in self._arguments:
                arguments[name] = _callback_configuration(mapping, xp)
        self._transform_active = True
        try:
            result = self.transform(
                **{name: arguments[name] for name in self._arguments}
            )
        finally:
            self._transform_active = False
        self._check_configuration()
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError('Transform must return (samples, validity)')
        return result

    def preview(
        self,
        inputs,
        *,
        completed_turn,
        time_seconds,
        valid=True,
        parameters=None,
        constants=None,
        record_ids=None,
        xp=None,
    ):
        """Preview a completed passage without advancing the window."""
        self._check_configuration()
        completed_turn = _positive_integer(completed_turn, 'completed_turn')
        time_seconds = float(time_seconds)
        if not math.isfinite(time_seconds) or time_seconds < 0:
            raise ValueError(
                'Acquisition completion time must be finite and nonnegative'
            )
        bound_xp = (
            self._source_context.nplike_lib
            if self._source_context is not None else self._xp
        )
        if xp is None:
            xp = np if bound_xp is None else bound_xp
        elif bound_xp is not None and xp is not bound_xp:
            raise ValueError('Acquisition array backend changed')
        state = replace(self._state)
        if state.last_boundary_turn is not None and (
            completed_turn <= state.last_boundary_turn
            or time_seconds <= state.last_boundary_time
        ):
            raise ValueError(
                'Acquisition completion turns and times must increase'
            )
        arrays = _arrays(inputs, xp, copy=False)
        schema = _schema(arrays)
        if state.raw_schema is not None and schema != state.raw_schema:
            raise ValueError(
                'Acquisition source names, shapes or dtypes changed')
        state.raw_schema = schema
        state.last_boundary_turn = completed_turn
        state.last_boundary_time = time_seconds
        invalidated = False
        elapsed_turns = completed_turn - self.first_sample_turn
        due = elapsed_turns >= 0 and elapsed_turns % self.sample_turns == 0
        latest_due = (
            self.first_sample_turn
            + (elapsed_turns // self.sample_turns) * self.sample_turns
        )
        if state.last_due_turn is not None:
            allowed_gap = self.sample_turns if due else 0
            if latest_due - state.last_due_turn > allowed_gap:
                self._clear(state)
                invalidated = True
                state.last_due_turn = (
                    latest_due - self.sample_turns if due else latest_due
                )
        if not due:
            return self._candidate(state, xp, False, invalidated)
        ids = (
            {name: completed_turn - 1 for name in arrays}
            if record_ids is None
            else dict(record_ids)
        )
        if set(ids) != set(arrays):
            raise ValueError('Record IDs must cover every acquisition source')
        for name, index in ids.items():
            if (
                not isinstance(index, numbers.Integral)
                or isinstance(index, bool)
                or index < 0
            ):
                raise TypeError('Record IDs must be nonnegative host integers')
            ids[name] = int(index)
        previous = state.highest_record_ids or dict.fromkeys(ids, -1)
        fresh = all(index > previous[name] for name, index in ids.items())
        state.last_due_turn = completed_turn
        # Keep the highest observed index even when a gap clears the window.
        state.highest_record_ids = {
            name: max(index, previous[name]) for name, index in ids.items()
        }
        usable = _validity(valid, xp)
        for value in arrays.values():
            usable = usable & xp.all(xp.isfinite(value))
        if not bool(usable) or not fresh:
            self._clear(state)
            return self._candidate(state, xp, False, True)
        # Copy the deque only when retaining a new passage. Existing arrays
        # remain owned and shared by committed states and pending previews.
        window = state.window.copy()
        window.append(_Passage(
            completed_turn, time_seconds,
            {name: value.copy() for name, value in arrays.items()},
            ids,
        ))
        state.window = window
        if len(window) < self.window_samples:
            return self._candidate(state, xp, False, invalidated)
        reduced = {}
        for name in arrays:
            stacked = xp.stack([passage.values[name] for passage in window])
            # Sum weighted moments before taking a ratio in the transform.
            reduced[name] = (
                xp.sum(stacked, axis=0)
                if self.reduction == 'sum'
                else xp.mean(stacked, axis=0)
            )
        if self.transform is None:
            samples, transformed_valid = reduced, True
        else:
            samples, transformed_valid = self._transform_sample(
                reduced, window, parameters, constants, xp)
        samples = _arrays(samples, xp)
        schema = _schema(samples)
        if state.sample_schema is not None and schema != state.sample_schema:
            raise ValueError(
                'Acquisition sample names, shapes or dtypes changed')
        state.sample_schema = schema
        usable = _validity(transformed_valid, xp)
        for value in samples.values():
            usable = usable & xp.all(xp.isfinite(value))
        if not bool(usable):
            self._clear(state)
            return self._candidate(state, xp, False, True)
        state.latest_samples = samples
        state.latest_valid = True
        return self._candidate(state, xp, True, invalidated)

    def commit(self, candidate):
        """Accept a preview from this acquisition's current generation."""
        self._check_configuration()
        if (
            candidate._owner is not self._owner
            or candidate._generation != self._generation
        ):
            raise ValueError('Stale or foreign acquisition candidate')
        self._state = candidate._state
        self._xp = candidate._xp
        self._generation += 1

    def step(self, inputs, **kwargs):
        """Preview and commit one acquisition update."""
        candidate = self.preview(inputs, **kwargs)
        self.commit(candidate)
        return candidate
