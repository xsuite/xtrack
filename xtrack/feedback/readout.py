# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

"""Read completed passages from tracking monitors on their own context."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from xobjects import ContextCpu, ContextCupy

from ..monitors.beam_stats_monitor.beam_stats_monitor import (
    BeamStatsMonitor, _COORDS,
)
from ..monitors.particles_monitor import ParticlesMonitor
from ..view import View


class IncompleteRecordError(ValueError):
    """The caller has not certified completion of this source passage."""


class SourceInvalidatedError(RuntimeError):
    """A clear or a change of frame or storage invalidated the source."""


@dataclass(frozen=True)
class MonitorRecord:
    """An owned measurement and its per-entry validity on the source context."""

    turn: int
    record_index: int
    value: object
    valid: object


@dataclass(frozen=True)
class BeamStatsRecord(MonitorRecord):
    """A statistical record with bin weights and one row-wide touched flag."""

    weight: object
    touched: object


def bind_monitor(monitor, quantity, *, bunch_slot=None):
    """Bind a coordinate or statistic from a supported tracking monitor."""
    if type(monitor) is View:
        monitor = monitor._get_viewed_object()
    if isinstance(monitor, BeamStatsMonitor):
        return _BeamStatsSource(monitor, quantity, bunch_slot=bunch_slot)
    if isinstance(monitor, ParticlesMonitor):
        if bunch_slot is not None:
            raise ValueError('Select particle IDs through ParticlesMonitor')
        return _ParticlesMonitorSource(monitor, quantity)
    raise TypeError(f'Unsupported monitor {type(monitor).__name__}')


class _TurnMonitorSource:
    """Shared binding and completion checks for full-turn monitor adapters."""

    requires_cycle_origin = False
    _record_stride = 1

    def __init__(self, monitor, quantity):
        if not isinstance(monitor._context, (ContextCpu, ContextCupy)):
            raise NotImplementedError('Monitor sources support CPU and CuPy')
        self._monitor = monitor
        self._quantity = str(quantity)
        self._bound_context = monitor._context
        self._bound_buffer = monitor._buffer
        self._bound_storage = monitor._buffer.buffer
        self._bound_offset = monitor._offset

        # Read the frame header in one transfer, using the native field offsets.
        # The structured dtype skips unrelated fields within that byte range.
        xobject = monitor._xobject
        fields = [getattr(type(xobject), name).get_offset(xobject)
                  for name in self._frame_fields]
        self._frame_offset = min(offset for _, offset in fields)
        self._frame_dtype = np.dtype(dict(
            names=self._frame_fields,
            formats=[field_type._dtype for field_type, _ in fields],
            offsets=[offset - self._frame_offset for _, offset in fields],
        ))
        self._frame = self._frame_signature()

    def _frame_signature(self):
        header = self._bound_buffer.to_bytearray(
            self._frame_offset, self._frame_dtype.itemsize)
        frame = np.frombuffer(header, dtype=self._frame_dtype, count=1)
        return frame[0].tolist()

    @property
    def monitor(self):
        """Return the bound monitor."""
        return self._monitor

    @property
    def quantity(self):
        """Return the quantity selected at binding."""
        return self._quantity

    @property
    def context(self):
        """Return the context holding the monitor data."""
        return self._bound_context

    @property
    def record_stride(self):
        """Return the spacing between recorded turns."""
        return self._record_stride

    def validate_binding(self):
        """Verify that the bound frame and cached storage are still current."""
        monitor = self.monitor
        if (
            monitor._context is not self._bound_context
            or monitor._buffer is not self._bound_buffer
            or monitor._offset != self._bound_offset
            or monitor._buffer.buffer is not self._bound_storage
            or self._frame_signature() != self._frame
        ):
            raise SourceInvalidatedError('Monitor changed; bind a new source')

    def validate_window(self, start_turn, stop_turn, *, sample_turns=1):
        """Check that requested samples fit the frame and are unwritten."""
        self.validate_binding()
        start, stop = self._frame[:2]
        if not start <= start_turn < stop_turn <= stop:
            raise ValueError(
                'Feedback monitor frame does not cover the requested turns'
            )
        stride = self.record_stride
        if (start_turn - start) % stride or sample_turns % stride:
            raise ValueError('Acquisition cadence must match the source grid')
        rows = slice((start_turn - start) // stride,
                     (stop_turn - start + stride - 1) // stride,
                     sample_turns // stride)
        if self._has_records(rows):
            raise ValueError(
                'Feedback monitor already contains the requested passages; '
                'use fresh records before tracking'
            )

    def _record_index(self, turn, completed_turn):
        for name, value in [('turn', turn), ('completed_turn', completed_turn)]:
            if not isinstance(value, Integral) or isinstance(value, bool):
                raise TypeError(f'{name} must be a host integer turn index')
        if turn >= completed_turn:
            raise IncompleteRecordError(
                f'Record turn {turn} has not completed at turn {completed_turn}'
            )
        self.validate_binding()
        # Compute the row without allocating the full monitor turn grid.
        start, stop = self._frame[:2]
        stride = self.record_stride
        if turn < start or turn >= stop:
            raise IndexError(
                f'Turn {turn} is outside monitor frame [{start}, {stop})')
        if (turn - start) % stride:
            raise IndexError(f'Turn {turn} is not recorded by stride {stride}')
        return (turn - start) // stride

    def read(self, turn, *, completed_turn):
        """Copy the value from a completed turn on its context."""
        return self.read_record(turn, completed_turn=completed_turn).value


class _BeamStatsSource(_TurnMonitorSource):
    """Bind a monitor and select one primitive sum, count or weighted mean.

    Parameters
    ----------
    monitor : BeamStatsMonitor
        Whole-beam, bunch or bunch-slice monitor after tracker construction.
    quantity : str
        ``num_particles``, ``sum_beta0_gamma0`` or ``sum_COORD``/``mean_COORD``
        for x, px, y, py, zeta, delta or pzeta. Request the corresponding
        statistic when constructing the monitor.
    bunch_slot : int or None
        Select one physical slot or retain the monitor's selected-slot axis.
    """

    _frame_fields = (
        'start_at_turn', 'stop_at_turn', 'every_n_turns', '_mode',
        '_num_records', '_num_selected_slots', '_num_slices',
    )

    def __init__(self, monitor, quantity, bunch_slot=None):
        super().__init__(monitor, quantity)
        self._bunch_slot = bunch_slot
        _, _, self._record_stride, mode, records, slots, slices = self._frame
        if mode not in (0, 1, 2):
            raise NotImplementedError(
                'Coasting source completion needs a different contract'
            )
        self.requires_cycle_origin = mode in (1, 2)
        if self.quantity in ('num_particles', 'sum_beta0_gamma0'):
            field = self.quantity
            self._is_mean = False
        elif self.quantity.startswith(('sum_', 'mean_')):
            prefix, coord = self.quantity.split('_', 1)
            if coord not in _COORDS:
                raise ValueError(f'Unsupported readout quantity {quantity!r}')
            field = f'sum_{coord}'
            self._is_mean = prefix == 'mean'
        else:
            raise ValueError(f'Unsupported readout quantity {quantity!r}')
        primitive = getattr(monitor.data, field)
        if len(primitive) == 0:
            raise ValueError(
                f'Primitive {field!r} was not allocated; request the '
                'corresponding public statistic when constructing the monitor'
            )
        self._slot_index = None
        if bunch_slot is not None:
            if (
                isinstance(bunch_slot, bool)
                or not isinstance(bunch_slot, Integral)
            ):
                raise TypeError('bunch_slot must be a physical integer slot')
            if mode == 0:
                raise ValueError('bunch_slot is not available in beam mode')
            self._slot_index = monitor.slot_index(bunch_slot)
        generations = getattr(
            self._bound_buffer, '_beam_stats_clear_generations', None)
        if generations is None:
            generations = {}
            self._bound_buffer._beam_stats_clear_generations = generations
        self._generations = generations
        self._bound_generation = generations.setdefault(self._bound_offset, 0)
        # Derive the row axes from native fields so Xobject aliases can bind.
        self._shape = (records,)
        if mode in (1, 2):
            self._shape += (slots,)
        if mode == 2:
            self._shape += (slices,)
        # Reuse context views while their frame and allocation remain bound.
        self._weight_data = monitor.data.num_particles
        self._primitive_data = primitive
        self._touched_data = monitor.touched_records.value

    @property
    def bunch_slot(self):
        """Return the selected physical slot, or None for all selected slots."""
        return self._bunch_slot

    @property
    def shape(self):
        """Return the selected row shape after any slot selection."""
        return self._shape[1:] if self._slot_index is None else self._shape[2:]

    def validate_binding(self):
        """Check the storage and detect clears through another wrapper."""
        super().validate_binding()
        if self._generations[self._bound_offset] != self._bound_generation:
            raise SourceInvalidatedError('Monitor cleared; bind a new source')

    def _has_records(self, rows):
        xp = self.context.nplike_lib
        weights = self._weight_data.reshape(self._shape)[rows]
        touched = self._touched_data[rows]
        return bool(xp.any(weights != 0) | xp.any(touched != 0))

    def _select(self, flat, index):
        shaped = flat.reshape(self._shape)
        # Slice before reshaping to keep even scalar results on the context.
        row = shaped[index : index + 1]
        if self._slot_index is not None:
            row = row[:, self._slot_index : self._slot_index + 1]
        return row.reshape(self.shape).copy()

    def _value(self, index, weight=None):
        primitive = self._select(self._primitive_data, index)
        if not self._is_mean:
            return primitive
        xp = self.context.nplike_lib
        if weight is None:
            weight = self._select(self._weight_data, index)
        positive_weight = xp.isfinite(weight) & (weight > 0)
        return xp.asarray(
            xp.where(
                positive_weight,
                primitive / xp.where(positive_weight, weight, 1.0),
                xp.nan,
            )
        )

    def read_record(self, turn, *, completed_turn):
        """Copy the value, weight and validity for one completed turn.

        The caller certifies completion through ``completed_turn - 1``.
        """
        index = self._record_index(turn, completed_turn)
        xp = self.context.nplike_lib
        weight = self._select(self._weight_data, index)
        value = self._value(index, weight)
        touched = self._touched_data[index : index + 1].reshape(()).copy()
        valid = xp.asarray(
            xp.isfinite(weight) & (weight > 0) & xp.isfinite(value))
        return BeamStatsRecord(int(turn), index, value, valid, weight, touched)

    def read(self, turn, *, completed_turn):
        """Copy the value from a completed turn on its context."""
        return self._value(self._record_index(turn, completed_turn))


class _ParticlesMonitorSource(_TurnMonitorSource):
    """Read native coordinates for one passage in particle-ID order."""

    _frame_fields = (
        'start_at_turn', 'stop_at_turn', 'part_id_start', 'part_id_end',
        'n_repetitions', 'ebe_mode',
    )

    def __init__(self, monitor, quantity):
        super().__init__(monitor, quantity)
        start, stop, first_id, last_id, repetitions, ebe_mode = self._frame
        if repetitions != 1 or ebe_mode:
            raise NotImplementedError(
                'Feedback needs a single full-turn ParticlesMonitor frame'
            )
        if self.quantity not in _COORDS:
            raise ValueError(f'Unsupported readout quantity {quantity!r}')
        if start >= stop or first_id >= last_id:
            raise ValueError('Particle source needs a valid frame and ID range')

        data_shape = (last_id - first_id, stop - start)
        xp = self.context.nplike_lib
        self._values = xp.asarray(
            getattr(monitor.data, self.quantity)).reshape(data_shape)
        self._states = monitor.data.state.reshape(data_shape)

    @property
    def shape(self):
        """Return the particle-ID axis length, including missing IDs."""
        return self._values.shape[:1]

    def _has_records(self, rows):
        xp = self.context.nplike_lib
        return bool(xp.any(self._states[:, rows] != 0))

    def read_record(self, turn, *, completed_turn):
        """Copy the coordinate vector and flag recorded, surviving particles."""
        index = self._record_index(turn, completed_turn)
        value = self._values[:, index].copy()
        # Recorded coordinates can be zero. State distinguishes unwritten IDs.
        xp = self.context.nplike_lib
        valid = (self._states[:, index] > 0) & xp.isfinite(value)
        return MonitorRecord(int(turn), index, value, valid)
