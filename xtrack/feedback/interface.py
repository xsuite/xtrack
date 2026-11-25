# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

"""Configure feedback elements and run them at completed line traversals."""

from dataclasses import asdict, dataclass
from functools import partial
import keyword
import math
from numbers import Integral

import numpy as np
import xobjects as xo
from scipy.constants import c as clight

from ..base_element import BeamElement, Replica
from ..composer.composer import Place, _flatten_components
from ..particles import Particles
from ..table import Table
from .acquisition import Acquisition
from .hardware import PhysicalActuator
from .output import KnobOutput, _check_time_dependent_geometry
from .program import _Program
from .readout import bind_monitor
from .runtime import FeedbackController, FeedbackSystem
from .timing import (
    _CLOCK_BATCH, _check_ramp_window, _check_turn_clock_resolution,
    _has_element_energy_change,
)


@dataclass(frozen=True, slots=True)
class MonitorInput:
    """A named coordinate or statistic from a placed monitor."""

    name: str
    monitor: str
    quantity: str
    unit: str = ''
    bunch_slot: int | None = None


@dataclass(frozen=True, slots=True)
class Output:
    """A named command bound to a physical actuator or Xdeps knobs."""

    name: str
    target: str | tuple[str, ...] | None = None
    actuator: PhysicalActuator | None = None
    unit: str = ''


@dataclass(frozen=True, slots=True)
class Parameter:
    """A named numerical parameter with immutable value and declared unit."""

    name: str
    value: np.ndarray
    unit: str = ''


@dataclass(frozen=True, slots=True)
class StateVariable:
    """A retained numerical state variable, with its initial value and unit."""

    name: str
    initial_value: np.ndarray
    unit: str = ''


def _immutable_array(value, dtype):
    """Own a finite numerical array with immutable backing storage."""
    result = np.asarray(value, dtype=dtype)
    if result.dtype.kind not in 'biufc' or not np.all(np.isfinite(result)):
        raise ValueError('Feedback configuration must be finite numerical data')
    return np.frombuffer(result.tobytes(), dtype=result.dtype).reshape(result.shape)


def _serialize_array(value):
    """Encode the array values for Xtrack's JSON writer."""
    if np.iscomplexobj(value):
        return {'real': value.real.tolist(), 'imag': value.imag.tolist()}
    return value.tolist()


def _deserialize_array(value, dtype, shape):
    if np.dtype(dtype).kind == 'c' and isinstance(value, dict):
        real = np.asarray(value['real'])
        imag = np.asarray(value['imag'])
        if real.shape != imag.shape:
            raise ValueError('Complex feedback components must have matching shapes')
        value = np.empty(real.shape, dtype=dtype)
        value.real[...] = real
        value.imag[...] = imag
    return np.asarray(value, dtype=dtype).reshape(shape) if shape is not None else value


class _Bindings:
    """Named configuration records shared by the four binding collections."""

    def __init__(self, feedback):
        self._feedback = feedback
        self._items = {}

    def _add(self, item):
        name = item.name
        if self._feedback._execution is not None:
            raise RuntimeError('Feedback configuration is frozen after preparation')
        if (
            not isinstance(name, str)
            or not name.isidentifier()
            or keyword.iskeyword(name)
            or name.startswith('_')
            or hasattr(type(self), name)
        ):
            raise ValueError(f'Invalid feedback binding name {name!r}')
        if name in self._items:
            raise ValueError(f'Duplicate feedback binding {name!r}')
        self._items[name] = item
        return item

    def __getitem__(self, name):
        return self._items[name]

    def __getattr__(self, name):
        try:
            return self._items[name]
        except KeyError:
            raise AttributeError(name) from None

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)

    def items(self):
        """Return the binding names and their configuration records."""
        return self._items.items()

    def get_table(self):
        """Inspect the names, binding classes and declared physical units."""
        return Table(
            {
                'name': np.array(list(self._items), dtype=str),
                'type': np.array(
                    [type(item).__name__ for item in self._items.values()]
                ),
                'unit': np.array([item.unit for item in self._items.values()]),
            }
        )

    def __repr__(self):
        return '\n'.join(repr(item) for item in self._items.values())


class Inputs(_Bindings):
    def new(self, *, name, monitor, quantity, unit='', bunch_slot=None):
        """Bind a monitor coordinate or statistic, optionally for one bunch."""
        if not isinstance(monitor, str) or not isinstance(quantity, str):
            raise TypeError('Monitor and quantity must be names')
        return self._add(MonitorInput(name, monitor, quantity, unit, bunch_slot))


class Outputs(_Bindings):
    def new(self, *, name, target=None, actuator=None, unit=''):
        """Bind a command to an explicit actuator or one or more Xdeps knobs."""
        if (target is None) == (actuator is None):
            raise ValueError('Specify exactly one output target or physical actuator')
        if actuator is not None and not isinstance(actuator, PhysicalActuator):
            raise TypeError('actuator must be a PhysicalActuator')
        if target is not None and not isinstance(target, str):
            target = tuple(target)
        return self._add(Output(name, target, actuator, unit))


class Parameters(_Bindings):
    def new(self, *, name, value, unit='', dtype=None):
        """Declare an owned parameter with a fixed numerical shape and dtype."""
        return self._add(Parameter(name, _immutable_array(value, dtype), unit))


class State(_Bindings):
    def new(self, *, name, initial_value, unit='', dtype=None):
        """Declare the initial value and dtype of a retained state variable."""
        value = _immutable_array(initial_value, dtype)
        return self._add(StateVariable(name, value, unit))

    def read(self, name):
        """Copy the current value on the tracking context, or the initial value on CPU."""
        runtime = self._feedback._runtime
        if runtime is None:
            return self[name].initial_value.copy()
        return runtime.memory[name].copy()


class Feedback(BeamElement):
    """A numerical controller placed at the end of a line.

    Declare inputs, parameters, retained state and one scalar or vector output.
    Set ``program`` to numerical assignments or a callable following the
    ``FeedbackController`` program contract. A callable program can use an
    ``Acquisition`` transform as ``acquisition_transform`` to process each
    completed sample before it enters the controller history. Configure
    numerical settings in ``parameters``.
    Binding units are labels.
    Place the controller in the terminal feedback block and prepare that block
    with ``line.feedback.enable()``. ``active`` selects which controllers in the
    group calculate. ``enabled`` reports whether this controller may calculate.

    Set the cadence with ``update_turns`` or ``update_frequency_hz``. By default
    the first calculation is due one interval after the tracking epoch.
    Processing and command transport delays are in seconds.

    Feedback runs at each completed forward turn, including during an active
    ``EnergyProgram``. Element-driven reference-energy changes are not
    supported. Commands affect subsequent transport. Array calculations use
    the CPU or CuPy tracking context.
    After total beam loss, the bound clock and actuator continue to advance
    with invalid measurements.
    A native pipeline hold interrupts the run. Copy the line before the first
    run. The copy shares the Python callables and starts with fresh tracking
    state. Use pure numerical callbacks. Return measurements, commands and
    state through the callback results so the runtime can commit them.
    Rebind captured context arrays when changing the tracking context.
    Dictionary and JSON export require a configuration without Python
    callables.
    """

    _xofields = {'_active': xo.Int8}
    iscollective = True
    allow_track = False
    allow_rot_and_shift = False
    needs_cpu = False

    def __init__(
        self,
        *,
        name,
        active=True,
        program=None,
        acquisition_transform=None,
        update_turns=None,
        update_frequency_hz=None,
        history_length=1,
        processing_delay_seconds=0.0,
        command_delay_seconds=0.0,
        log_capacity=1024,
        first_calculation_turn=None,
        first_calculation_time_seconds=None,
        **kwargs,
    ):
        if not isinstance(name, str) or not name:
            raise ValueError('Feedback needs a nonempty name')
        if not isinstance(active, bool):
            raise TypeError('active must be Boolean')
        super().__init__(_active=int(active), **kwargs)
        self.name = name
        self.program = program
        self.acquisition_transform = acquisition_transform
        self.update_turns = update_turns
        self.update_frequency_hz = update_frequency_hz
        self.history_length = history_length
        self.processing_delay_seconds = processing_delay_seconds
        self.command_delay_seconds = command_delay_seconds
        self.log_capacity = log_capacity
        self.first_calculation_turn = first_calculation_turn
        self.first_calculation_time_seconds = first_calculation_time_seconds
        self._runtime = None
        self._execution = None
        self.inputs = Inputs(self)
        self.outputs = Outputs(self)
        self.parameters = Parameters(self)
        self.state = State(self)

    @property
    def active(self):
        """Whether this controller is selected when its line group is enabled."""
        return bool(self._active)

    @active.setter
    def active(self, value):
        if not isinstance(value, bool):
            raise TypeError('active must be Boolean')
        if value:
            self.enable()
        else:
            self.disable()

    def enable(self, *, state=None):
        """Select this controller, reconciling state after cancelled commands."""
        if self._runtime is not None:
            self._runtime.enable(memory=state)
            if not self._execution.enabled:
                self._runtime.disable(cancel_pending=False)
        elif state is not None:
            raise ValueError('State reconciliation requires a prepared runtime')
        self._active = 1

    def disable(self, *, cancel_pending=False):
        """Pause calculations, optionally dropping queued commands."""
        if self._runtime is not None:
            self._runtime.disable(cancel_pending=cancel_pending)
        self._active = 0

    @property
    def log(self):
        """Return the retained boundary records in chronological order."""
        return () if self._runtime is None else tuple(self._runtime.log)

    @property
    def enabled(self):
        """Whether the selected controller belongs to an enabled, usable group."""
        execution = self._execution
        return bool(
            self.active
            and execution is not None
            and execution.enabled
            and execution.failure is None
        )

    def track(self, particles):
        """Process the terminal feedback block during collective tracking."""
        if self._execution is None:
            if self.active:
                raise RuntimeError(
                    f'Feedback {self.name!r}: call line.feedback.enable()'
                )
            return
        self._execution.finish_traversal(self, particles)

    def _settings(self):
        """Snapshot scalar settings that must stay fixed after preparation."""
        return (
            self.name,
            self.program if isinstance(self.program, str) else id(self.program),
            id(self.acquisition_transform),
            self.update_turns,
            self.update_frequency_hz,
            self.history_length,
            self.processing_delay_seconds,
            self.command_delay_seconds,
            self.log_capacity,
            self.first_calculation_turn,
            self.first_calculation_time_seconds,
        )

    def _controller_options(self, line, first_feedback_index):
        """Bind this controller's program and channels to the prepared line."""
        program = self.program
        if not (isinstance(program, str) or callable(program)):
            raise TypeError(
                f'Feedback {self.name!r} needs program text or a callable'
            )
        if isinstance(program, str) and self.acquisition_transform is not None:
            raise ValueError(
                'An acquisition transform requires a callable program'
            )
        if not self.inputs or len(self.outputs) != 1:
            raise ValueError('Feedback needs inputs and exactly one named output')
        if (
            not isinstance(self.history_length, Integral)
            or isinstance(self.history_length, bool)
            or self.history_length < 1
        ):
            raise ValueError('history_length must be a positive integer')
        sources = {}
        for name, binding in self.inputs.items():
            if line.element_names.count(binding.monitor) != 1:
                raise ValueError(f'Monitor {binding.monitor!r} must occur exactly once')
            if line.element_names.index(binding.monitor) >= first_feedback_index:
                raise ValueError(
                    'Feedback inputs must precede the terminal feedback block'
                )
            source = bind_monitor(
                line.get(binding.monitor),
                binding.quantity,
                bunch_slot=binding.bunch_slot,
            )
            if source.requires_cycle_origin:
                source_s = line._get_s_position(binding.monitor)
                if source_s != 0:
                    raise NotImplementedError(
                        'Bunch and slice feedback require pickups at the cycle origin; '
                        'general bunch-arrival completion is not implemented'
                    )
            sources[name] = source
        context = next(iter(sources.values())).context
        xp = context.nplike_lib
        _, output_binding = next(iter(self.outputs.items()))
        if output_binding.actuator is None:
            output = KnobOutput(line, output_binding.target)
        else:
            owner = output_binding.actuator.owner
            if owner is not None and getattr(owner, 'env', owner) is not line.env:
                raise ValueError('Physical actuator belongs to a different environment')
            output = output_binding.actuator.copy_for(line)
        field_output = (
            output._bound_output() if isinstance(output, PhysicalActuator) else output
        )
        field_output._check_reference_fields()
        parameters = {
            name: item.value.item()
            if item.value.shape == ()
            else xp.asarray(item.value)
            for name, item in self.parameters.items()
        }
        state = {
            name: xp.asarray(item.initial_value) for name, item in self.state.items()
        }
        if isinstance(program, str):
            acquired_names = set(sources) | {
                'applied_output', 'held_target', 'pending_output',
            }
            if isinstance(output, PhysicalActuator):
                acquired_names.update(('total_target', 'nominal_target'))
            if self.history_length > 1:
                acquired_names.update(name + '_history' for name in sources)
            program = _Program(
                program, name=self.name, acquired=acquired_names,
                parameters=parameters, state=state,
                output=output_binding.name, xp=xp,
            )
        return dict(
            name=self.name,
            acquisition=Acquisition(
                sources=sources, transform=self.acquisition_transform,
            ),
            program=program,
            output=output,
            parameters=parameters,
            memory=state,
            history_length=self.history_length,
            update_turns=self.update_turns,
            update_frequency_hz=self.update_frequency_hz,
            processing_delay_seconds=self.processing_delay_seconds,
            command_delay_seconds=self.command_delay_seconds,
            log_capacity=self.log_capacity,
            output_key=output_binding.name,
            first_calculation_turn=self.first_calculation_turn,
            first_calculation_time_seconds=self.first_calculation_time_seconds,
        )

    def to_dict(self, **kwargs):
        """Export a cold feedback configuration with a text program."""
        if callable(self.program) or self.acquisition_transform is not None:
            raise NotImplementedError(
                f'Cannot serialize feedback {self.name!r} with Python callables'
            )
        return self._configuration_dict()

    def _configuration_dict(self):
        """Build the cold configuration for text export and in-memory copies."""
        if self._runtime is not None:
            raise NotImplementedError(
                'Copy or serialize feedback before the first tracking run'
            )
        return dict(
            __class__=type(self).__name__,
            name=self.name,
            active=self.active,
            program=None if callable(self.program) else self.program,
            update_turns=self.update_turns,
            update_frequency_hz=self.update_frequency_hz,
            history_length=self.history_length,
            processing_delay_seconds=self.processing_delay_seconds,
            command_delay_seconds=self.command_delay_seconds,
            log_capacity=self.log_capacity,
            first_calculation_turn=self.first_calculation_turn,
            first_calculation_time_seconds=self.first_calculation_time_seconds,
            inputs=[asdict(item) for _, item in self.inputs.items()],
            parameters=[
                dict(
                    name=name,
                    value=_serialize_array(item.value),
                    dtype=item.value.dtype.str,
                    shape=list(item.value.shape),
                    unit=item.unit,
                )
                for name, item in self.parameters.items()
            ],
            state=[
                dict(
                    name=name,
                    initial_value=_serialize_array(item.initial_value),
                    dtype=item.initial_value.dtype.str,
                    shape=list(item.initial_value.shape),
                    unit=item.unit,
                )
                for name, item in self.state.items()
            ],
            outputs=[
                dict(
                    name=name,
                    target=item.target,
                    unit=item.unit,
                    actuator=(
                        None if item.actuator is None else item.actuator.to_dict()
                    ),
                )
                for name, item in self.outputs.items()
            ],
        )

    @classmethod
    def from_dict(cls, data, **kwargs):
        """Restore feedback configuration for a fresh tracking run."""
        data = dict(data)
        data.pop('__class__', None)
        groups = {
            name: data.pop(name, [])
            for name in ['inputs', 'parameters', 'state', 'outputs']
        }
        result = cls(**data, **kwargs)
        for group, items in groups.items():
            for item in items:
                item = dict(item)
                if group == 'outputs' and item.get('actuator') is not None:
                    item['actuator'] = PhysicalActuator(**item['actuator'])
                if group in ('parameters', 'state'):
                    key = 'value' if group == 'parameters' else 'initial_value'
                    item[key] = _deserialize_array(
                        item[key], item.get('dtype'), item.pop('shape', None))
                getattr(result, group).new(**item)
        return result

    def copy(self, **kwargs):
        """Copy cold configuration, keeping the same live Python callables."""
        result = self.from_dict(self._configuration_dict(), **kwargs)
        result.program = self.program
        result.acquisition_transform = self.acquisition_transform
        return result


class FeedbackLine:
    """Discover and enable feedback elements already placed in a Line."""

    def __init__(self, line):
        self.line = line

    @property
    def enabled(self):
        """Whether this line's feedback group is prepared and enabled."""
        execution = getattr(self.line, '_feedback_execution', None)
        return bool(
            execution is not None and execution.enabled and execution.failure is None
        )

    @enabled.setter
    def enabled(self, value):
        if not isinstance(value, bool):
            raise TypeError('enabled must be Boolean')
        if value:
            self.enable()
        else:
            self.disable()

    def _placement_names(self):
        """Read element names before tracker construction can move buffers."""
        line = self.line
        if line.mode != 'compose' or line._has_valid_tracker():
            return line.element_names

        if not any(isinstance(element, Feedback)
                   for element in line._element_dict.values()):
            return ()
        composer = line.composer
        components = _flatten_components(
            line.env, composer.components, refer=composer.refer,
        )
        return [item.name if isinstance(item, Place) else item for item in components]

    def _placed_feedbacks(self):
        placed = []
        for name in self._placement_names():
            element = self.line._element_dict[name]
            if isinstance(element, Feedback):
                placed.append((name, element))
        return placed

    def _check_placement(self, parts=None, *, require_enabled=False):
        """Protect placed feedback ownership before tracker setup or traversal."""
        line = self.line
        if parts is None:
            parts = (line._element_dict[name] for name in self._placement_names())

        execution = getattr(line, '_feedback_execution', None)
        for part in parts:
            if isinstance(part, Replica) and isinstance(part.resolve(line), Feedback):
                raise NotImplementedError(
                    'Tracking a Replica of a placed feedback is unsupported')
            if not isinstance(part, Feedback):
                continue
            if part._execution is not None and part._execution is not execution:
                raise RuntimeError('A placed feedback is prepared on another line')
            if require_enabled and execution is None and part.active:
                raise RuntimeError(
                    'Call line.feedback.enable() before tracking feedbacks')

    def __getitem__(self, name):
        """Look up a controller by its name within the feedback group."""
        matches = [fb for _, fb in self._placed_feedbacks() if fb.name == name]
        if len(matches) != 1:
            raise KeyError(name)
        return matches[0]

    def get_table(self):
        """List controller names, line element names, cadence and activation."""
        elements = self._placed_feedbacks()
        return Table(
            {
                'name': np.array([fb.name for _, fb in elements]),
                'element': np.array([name for name, _ in elements]),
                'update_turns': np.array([fb.update_turns for _, fb in elements]),
                'update_frequency_hz': np.array([
                    fb.update_frequency_hz for _, fb in elements
                ]),
                'active': np.array([fb.active for _, fb in elements]),
                'enabled': np.array([fb.enabled for _, fb in elements]),
            }
        )

    def enable(self):
        """Prepare or resume the feedback group at the end of the line."""
        line = self.line
        existing = getattr(line, '_feedback_execution', None)
        if existing is not None:
            existing.check_configuration()
            if any(
                fb.active
                and fb._runtime is not None
                and fb._runtime._requires_reconciliation
                for fb in existing.feedbacks
            ):
                raise RuntimeError(
                    'Reconcile cancelled controller state before enabling the group'
                )
            for fb in existing.feedbacks:
                if fb.active and fb._runtime is not None:
                    fb._runtime.enable()
            existing.enabled = True
            return
        placed = self._placed_feedbacks()
        if not placed:
            self._check_placement()
            raise ValueError('No feedback elements are placed in this line')
        peers = []
        for sibling in line.env._lines_weakrefs:
            owner = getattr(sibling, '_feedback_execution', None)
            if owner is not None and owner.line is sibling:
                peers.append(owner)
        if not line._has_valid_tracker():
            # A different buffer would discard the prepared sibling trackers.
            buffer = peers[0].tracker._buffer if peers else None
            line.build_tracker(_buffer=buffer)
            placed = self._placed_feedbacks()
        if line.tracker.iscollective:
            self._check_placement(line.tracker._parts)
        first_feedback_index = line.element_names.index(placed[0][0])
        tail = list(line.element_names[first_feedback_index:])
        if tail[-1] == '_end_point':
            tail.pop()
        if tail != [name for name, _ in placed]:
            raise ValueError('Feedbacks must form a contiguous terminal block')
        feedbacks = [fb for _, fb in placed]
        if len({fb.name for fb in feedbacks}) != len(feedbacks):
            raise ValueError('Placed feedback names must be unique')
        execution = _LineFeedbackRun(line, feedbacks)
        execution.check_configuration()
        options = [
            fb._controller_options(line, first_feedback_index) for fb in feedbacks
        ]
        # Check runtime options and writer ownership before enabling the group.
        system = FeedbackSystem(
            [FeedbackController(**item) for item in options], lines=[line]
        )
        if peers:
            # Paused executions can still deliver commands and settle fields.
            system._check_ownership(
                item['output'] for peer in peers for item in peer.controller_options
            )
        execution.controller_options = options
        line._feedback_execution = execution
        for fb in feedbacks:
            fb._execution = execution

    def disable(self, *, cancel_pending=False):
        """Prepare placed feedbacks if needed, then pause their calculations.

        Acquisition and actuator response continue. Pending commands are
        retained unless ``cancel_pending`` is requested.
        """
        execution = getattr(self.line, '_feedback_execution', None)
        if execution is None and self._placed_feedbacks():
            self.enable()
            execution = self.line._feedback_execution
        if execution is not None:
            execution.enabled = False
            for fb in execution.feedbacks:
                if fb._runtime is not None:
                    fb._runtime.disable(cancel_pending=cancel_pending)


class _LineFeedbackRun:
    """Own the configuration, clock and runtime for one prepared line."""

    def __init__(self, line, feedbacks):
        self.line, self.feedbacks = line, feedbacks
        self.tracker = line.tracker
        self.configuration = self._configuration_signature()
        self.element_names = tuple(line.element_names)
        self.circumference = float(self.tracker._tracker_data_base.line_length)
        self.energy_program = (
            line.energy_program if line.enable_time_dependent_vars else None
        )
        self.enabled = True
        self.system = None
        self.particles = None
        self.failure = None
        self.controller_options = None
        self.epoch_turn = self.epoch_time = self.revolution_period = None
        self.p0c = None
        self._turn_start_time = None
        self._turn_start_reference_turns = None

    def check_configuration(self, *, boundary=False):
        """Check the prepared configuration and, at admission, live geometry."""
        if self.failure is not None:
            raise RuntimeError(
                'Feedback tracking failed; reconstruct a consistent run'
            ) from self.failure
        if self.line.tracker is not self.tracker:
            raise RuntimeError(
                'Feedback line/tracker changed; prepare a fresh configuration'
            )
        # Equal circumference alone does not guarantee unchanged pickup positions.
        if not boundary and (
            tuple(self.line.element_names) != self.element_names
            or not np.array_equal(
                self.line._get_s_elements(mode='downstream'),
                self.tracker._tracker_data_base._line_table.s[1:],
            )
        ):
            raise RuntimeError('Feedback line geometry changed after preparation')
        if self._configuration_signature() != self.configuration:
            raise RuntimeError('Feedback configuration changed after preparation')
        if getattr(self.tracker, '_sync_time_controller', None) is not None:
            raise NotImplementedError(
                'Feedback frame-clock SyncTime is not implemented'
            )
        program = (self.line.energy_program
                   if self.line.enable_time_dependent_vars else None)
        if program is not self.energy_program:
            raise RuntimeError('Feedback EnergyProgram changed after preparation')
        if program is not None and program.line is not self.line:
            raise ValueError('Feedback requires an EnergyProgram attached to its line')
        if not boundary:
            _check_time_dependent_geometry(self.line)
            if _has_element_energy_change(self.tracker):
                raise NotImplementedError(
                    'Feedback cannot use element-driven reference-energy changes; '
                    'use an EnergyProgram'
                )
        if self.line.skip_end_turn_actions or not self.line.reset_s_at_end_turn:
            raise ValueError('Feedback needs full end-turn clock actions and s reset')

    def _configuration_signature(self):
        return tuple(
            (
                fb._settings(),
                id(fb.inputs),
                id(fb.outputs),
                id(fb.parameters),
                id(fb.state),
            )
            for fb in self.feedbacks
        )

    def _read_clock(self, particles):
        """Read the native clock, inferring an unset reference-turn epoch."""
        time = float(particles.time_s)
        period = self.circumference / (float(particles._xobject.beta0[0]) * clight)
        raw_turn = float(particles.reference_turns)
        if math.isnan(raw_turn):
            raw_turn = float(self.tracker._get_reference_turns(particles))
        return time, period, raw_turn

    def validate_track(
        self, particles, *, ele_start, ele_stop, num_elements, backtrack,
        freeze_longitudinal, num_turns=None,
    ):
        """Check the full request before native setup or pickup writes."""
        self.check_configuration()
        if not isinstance(particles, Particles):
            raise NotImplementedError(
                'Placed feedback requires ordinary tracked Particles'
            )
        sources = self.controller_options[0]['acquisition'].sources
        monitor_context = next(iter(sources.values())).context
        if particles._context is not monitor_context:
            raise ValueError(
                'Feedback particles and monitor sources need the same context'
            )
        if (
            ele_start not in (None, 0, self.element_names[0])
            or particles.start_tracking_at_element >= 0
            or ele_stop is not None
            or num_elements is not None
            or backtrack
        ):
            raise NotImplementedError(
                'Prepared feedback requires forward complete-turn tracking'
            )
        if freeze_longitudinal:
            raise NotImplementedError(
                'Prepared feedback does not support freeze_longitudinal'
            )
        turns = 1 if num_turns is None else num_turns
        if not isinstance(turns, Integral) or isinstance(turns, bool) or turns < 1:
            raise ValueError('Feedback num_turns must be a positive integer')
        xp = particles._context.nplike_lib
        alive = particles.state > 0
        # A bound run still advances its clock and actuator after total loss.
        if self.system is None and not bool(xp.any(alive)):
            raise ValueError('Feedback needs surviving particles')
        if bool(xp.any(particles.at_element[alive] != 0)) or bool(
            xp.any(particles.s[alive] != 0)
        ):
            raise ValueError('Feedback tracking must begin at the line origin')
        time, period, raw_turn = self._read_clock(particles)
        if not math.isfinite(time) or time < 0:
            raise ValueError('Feedback requires a finite nonnegative clock epoch')
        if not math.isfinite(period) or period <= 0:
            raise ValueError('Feedback requires a positive finite revolution period')
        if not math.isfinite(raw_turn) or raw_turn < 0:
            raise ValueError('Feedback needs a finite nonnegative reference turn')
        turn = round(raw_turn)
        # Allow roundoff in the native time-to-turn conversion at the epoch.
        roundoff = max(1e-9, 2 * math.ulp(raw_turn))
        if not math.isclose(raw_turn, turn, abs_tol=roundoff, rel_tol=0):
            raise ValueError('Feedback needs an integer reference-turn epoch')
        if bool(xp.any(particles.at_turn[alive] != turn)):
            raise ValueError('Feedback particle and reference-turn epochs disagree')
        frequencies = [opts['update_frequency_hz'] for opts in self.controller_options
                       if opts['update_frequency_hz'] is not None]
        longest_period = period
        if self.energy_program is None:
            _check_turn_clock_resolution(time + turns * period, period)
        else:
            program = self.energy_program
            last_turn = turn + turns
            end = _check_ramp_window(program, turn, last_turn, time, raw_turn)
            longest_period = 0.
            for start in range(turn, last_turn, _CLOCK_BATCH):
                starts = np.arange(start, min(start + _CLOCK_BATCH, last_turn))
                periods = program._elapsed_time_between_turns(starts, starts + 1)
                _check_turn_clock_resolution(end, float(periods.min()))
                longest_period = max(longest_period, float(periods.max()))
        if any(
            frequency * longest_period > 1 + 128 * np.finfo(float).eps
            for frequency in frequencies
        ):
            raise NotImplementedError(
                'Full-turn feedback cannot execute faster than '
                'the revolution frequency'
            )
        expected_turn = (
            self.system._completed_turn
            if self.system is not None and self.system._completed_turn is not None
            else self.epoch_turn
        )
        if expected_turn is not None and turn != expected_turn:
            raise RuntimeError('Feedback clock/turn changed outside tracking')
        for opts in self.controller_options:
            for source in opts['acquisition'].sources.values():
                source.validate_window(turn, turn + turns)
        t_sim = float(particles.t_sim)
        if self.energy_program is None and t_sim and not math.isclose(
            t_sim, period, rel_tol=128 * np.finfo(float).eps, abs_tol=0
        ):
            raise ValueError('Feedback requires a fixed full-turn simulation period')
        p0c = float(particles._xobject.p0c[0])
        if self.system is not None:
            if self.energy_program is None:
                reference_changed = period != self.revolution_period or p0c != self.p0c
            else:
                reference_changed = bool(xp.any(alive & (particles.p0c != self.p0c)))
            if particles is not self.particles or reference_changed:
                raise RuntimeError('Feedback particles or reference energy changed')
            tolerance = 8 * np.spacing(max(abs(self._turn_start_time), period))
            if abs(time - self._turn_start_time) > tolerance:
                raise RuntimeError('Feedback particle clock changed outside tracking')
            if raw_turn != self._turn_start_reference_turns:
                raise RuntimeError('Feedback reference turns changed outside tracking')
            if (self.energy_program is not None
                    and t_sim != self.revolution_period):
                raise RuntimeError('Feedback reference period changed outside tracking')
            self.system._check_ownership()
            self.system._check_applied_outputs()

    def bind_runtime(self, particles):
        """Bind the beam and its clock after native session validation."""
        if self.system is not None:
            return
        time, period, raw_turn = self._read_clock(particles)
        turn = round(raw_turn)
        controllers = [FeedbackController(**opts) for opts in self.controller_options]
        elapsed = (period if self.energy_program is None else partial(
            self.energy_program._elapsed_time_between_turns, turn,
        ))
        self.system = FeedbackSystem(
            controllers, lines=[self.line], _turn_clock=(turn, time, elapsed),
        )
        if self.energy_program is not None:
            # Use the admitted integer epoch for native lookup at ramp knots.
            particles.reference_turns = raw_turn = turn
        self.epoch_turn, self.epoch_time, self.revolution_period = turn, time, period
        self.p0c = float(particles._xobject.p0c[0])
        self._turn_start_time = time
        self._turn_start_reference_turns = raw_turn
        self.particles = particles
        for fb, controller in zip(self.feedbacks, controllers):
            if not (fb.active and self.enabled):
                controller.disable()
            fb._runtime = controller

    def start_ramped_traversal(self, particles):
        """Capture the momentum and period installed by the native tracker."""
        self.p0c = float(self.line.particle_ref._xobject.p0c[0])
        self.revolution_period = float(particles.t_sim)

    def finish_traversal(self, feedback, particles):
        """Advance the group once, before Xtrack increments the turn clock."""
        # The first element processes the entire terminal feedback block.
        if feedback is not self.feedbacks[0]:
            return
        if particles is not self.particles or self.system is None:
            raise RuntimeError('Use ordinary line.track for prepared feedback')
        try:
            self.check_configuration(boundary=True)
            alive = particles.state > 0
            xp = particles._context.nplike_lib
            if self.energy_program is None:
                changed = float(particles._xobject.p0c[0]) != self.p0c
            else:
                # Lost particles keep the reference momentum at their loss.
                changed = bool(xp.any(alive & (particles.p0c != self.p0c)))
            if changed:
                raise RuntimeError(
                    'Feedback reference momentum changed within tracking'
                )
            period = self.revolution_period
            tolerance = 8 * np.spacing(max(abs(self._turn_start_time), period))
            if not math.isclose(
                float(particles.time_s), self._turn_start_time,
                rel_tol=0, abs_tol=tolerance,
            ):
                raise RuntimeError('Feedback clock changed within a traversal')
            if float(particles.reference_turns) != self._turn_start_reference_turns:
                raise RuntimeError('Feedback reference turns changed within tracking')
            if float(particles.t_sim) != period:
                raise RuntimeError('Feedback turn period changed within tracking')
            previous = self.system._completed_turn
            completed = (self.epoch_turn if previous is None else previous) + 1
            next_native_time = float(particles.time_s) + period
            time = (self.epoch_time + (completed - self.epoch_turn) * period
                    if self.energy_program is None else next_native_time)
            length = self.circumference
            tolerance = 128 * np.finfo(float).eps * max(abs(length), 1.0)
            if not bool(xp.all(xp.abs(particles.s[alive] - length) <= tolerance)):
                raise ValueError('Feedback boundary did not complete the full line')
            self.system.step(completed, time)
            # Match the next native end-turn addition, including its rounding.
            self._turn_start_time = next_native_time
            self._turn_start_reference_turns += 1
        except BaseException as error:
            self.failure = error
            raise
