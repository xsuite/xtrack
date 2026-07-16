import numpy as np
import xobjects as xo

from ..base_element import BeamElement


_COORDS = ('x', 'px', 'y', 'py', 'zeta', 'delta')
_PLANES = {
    'x': ('x', 'px'),
    'y': ('y', 'py'),
    'zeta': ('zeta', 'delta'),
}

_SECOND_MOMENTS = (
    'x_x', 'x_px', 'x_y', 'x_py', 'x_zeta', 'x_delta',
    'px_px', 'px_y', 'px_py', 'px_zeta', 'px_delta',
    'y_y', 'y_py', 'y_zeta', 'y_delta',
    'py_py', 'py_zeta', 'py_delta',
    'zeta_zeta', 'zeta_delta',
    'delta_delta',
)

_DEFAULT_STATS = (
    'num_particles',
    'mean_x', 'mean_y',
    'sigma_x', 'sigma_y',
)


class BeamStatsMonitorRecord(xo.HybridClass):
    _xofields = {
        'num_particles': xo.Float64[:],
        'sum_beta0_gamma0': xo.Float64[:],
        'sum_x': xo.Float64[:],
        'sum_px': xo.Float64[:],
        'sum_y': xo.Float64[:],
        'sum_py': xo.Float64[:],
        'sum_zeta': xo.Float64[:],
        'sum_delta': xo.Float64[:],
        'sum_x_x': xo.Float64[:],
        'sum_x_px': xo.Float64[:],
        'sum_x_y': xo.Float64[:],
        'sum_x_py': xo.Float64[:],
        'sum_x_zeta': xo.Float64[:],
        'sum_x_delta': xo.Float64[:],
        'sum_px_px': xo.Float64[:],
        'sum_px_y': xo.Float64[:],
        'sum_px_py': xo.Float64[:],
        'sum_px_zeta': xo.Float64[:],
        'sum_px_delta': xo.Float64[:],
        'sum_y_y': xo.Float64[:],
        'sum_y_py': xo.Float64[:],
        'sum_y_zeta': xo.Float64[:],
        'sum_y_delta': xo.Float64[:],
        'sum_py_py': xo.Float64[:],
        'sum_py_zeta': xo.Float64[:],
        'sum_py_delta': xo.Float64[:],
        'sum_zeta_zeta': xo.Float64[:],
        'sum_zeta_delta': xo.Float64[:],
        'sum_delta_delta': xo.Float64[:],
    }


class BeamStatsMonitorTouchedRecords(xo.HybridClass):
    _xofields = {
        'value': xo.Int64[:],
    }


class BeamStatsMonitor(BeamElement):
    """
    Monitor weighted beam statistics.

    The monitor records beam statistics over selected turns. It operates in
    one of three modes selected from the constructor inputs:

    - beam mode: no bunch or slice inputs are provided. One value is recorded
      per logged turn for the whole beam.
    - bunch mode: bunch inputs are provided without slice inputs. One value is
      recorded per logged turn and selected physical slot. Whole-beam
      statistics are also available.
    - slice mode: `zeta_range` and `num_slices` are provided. One value is
      recorded per logged turn, selected physical slot, and longitudinal slice.
      Per-bunch and whole-beam statistics are also available.

    All statistics are weighted by ``particles.weight``. The public
    ``num_particles`` quantity is therefore the sum of particle weights in each
    bin, not the number of macroparticles.

    Requested statistics are available as attributes at the most detailed
    recorded level, for example ``monitor.mean_x``. The :meth:`get` method
    gives access to a specific aggregation level and accepts physical
    selectors:

    .. code-block:: python

        monitor.get("mean_x")
        # Default level, with shape depending on the monitor mode.

        monitor.get("mean_x", level="beam")
        # Shape: (n_logged_turns,)

        monitor.get("mean_x", level="bunch")
        # Shape: (n_logged_turns, n_selected_slots)

        monitor.get("mean_x", level="bunch", slot=3)
        # Shape: (n_logged_turns,)

        monitor.get("mean_x", level="slice")
        # Shape: (n_logged_turns, n_selected_slots, n_slices)

        monitor.get("mean_x", level="slice", slot=3, slice_index=12)
        # Shape: (n_logged_turns,)

    Scalar selectors such as ``turn=10``, ``slot=3``, or ``slice_index=12``
    remove the selected axis by default. Use ``keepdims=True`` to preserve
    length-one axes.

    Parameters
    ----------
    start_at_turn : int, optional
        First turn to record, inclusive.
    stop_at_turn : int, optional
        Last turn to record, exclusive. If omitted, record only
        `start_at_turn`.
    every_n_turns : int, optional
        Record turns separated by this stride.
    zeta_range : tuple[float, float], optional
        Longitudinal range for slice mode. Must be provided together with
        `num_slices`.
    num_slices : int, optional
        Number of longitudinal slices per selected bunch.
    num_bunches : int, optional
        Number of consecutive filled slots when neither `filled_slots` nor
        `filling_scheme` is provided.
    filling_scheme : array_like, optional
        Boolean/integer filling scheme identifying filled physical slots.
    filled_slots : array_like, optional
        Explicit physical slot numbers which are filled.
    selected_slots : array_like, optional
        Filled physical slots to record. Output follows this order.
    bunch_spacing_zeta : float, optional
        Longitudinal spacing between adjacent physical slots.
    stats : sequence of str, optional
        Requested public statistics.
    output_file : str or path-like, optional
        HDF5 file where :meth:`save_to_file` appends the current frame.
    """

    _xofields = {
        'start_at_turn': xo.Int64,
        'stop_at_turn': xo.Int64,
        'every_n_turns': xo.Int64,
        '_mode': xo.Int64,
        '_num_records': xo.Int64,
        '_num_selected_slots': xo.Int64,
        '_num_slices': xo.Int64,
        '_z_min_edge': xo.Float64,
        '_dzeta': xo.Float64,
        '_bunch_spacing_zeta': xo.Float64,
        '_selected_slots': xo.Int64[:],
        '_filled_slots': xo.Int64[:],
        '_slot_to_selected': xo.Int64[:],
        'data': BeamStatsMonitorRecord,
        'touched_records': BeamStatsMonitorTouchedRecords,
    }

    _extra_c_sources = [
        '#include "xtrack/monitors/beam_stats_monitor.h"',
    ]

    behaves_like_drift = True
    allow_loss_refinement = True

    _RAW_FIELDS = ('num_particles', 'sum_beta0_gamma0',
                   *(f'sum_{coord}' for coord in _COORDS),
                   *(f'sum_{moment}' for moment in _SECOND_MOMENTS))

    def __init__(self, *,
                 start_at_turn=0,
                 stop_at_turn=None,
                 every_n_turns=1,
                 zeta_range=None,
                 num_slices=None,
                 num_bunches=1,
                 filling_scheme=None,
                 filled_slots=None,
                 selected_slots=None,
                 bunch_spacing_zeta=None,
                 stats=None,
                 output_file=None,
                 _xobject=None,
                 **kwargs):
        """
        Initialize the monitor configuration and primitive moment storage.
        """

        if _xobject is not None:
            super().__init__(_xobject=_xobject)
            self._output_file = None
            return

        slice_mode = zeta_range is not None or num_slices is not None
        if (zeta_range is None) != (num_slices is None):
            raise ValueError(
                '`zeta_range` and `num_slices` must be provided together')
        if stop_at_turn is None:
            stop_at_turn = start_at_turn + 1
        if every_n_turns <= 0:
            raise ValueError('`every_n_turns` must be positive')

        # Keep requested stats in user order while ignoring duplicates.
        raw_stats = _DEFAULT_STATS if stats is None else tuple(stats)
        stats = []
        for stat in raw_stats:
            if stat not in stats:
                stats.append(stat)
        stats = tuple(stats)
        _check_supported_stats(stats)

        # Bunch mode is selected by any bunch-related input, unless slice
        # inputs already selected the more detailed slice mode.
        bunch_mode = (
            not slice_mode
            and (filled_slots is not None
                 or filling_scheme is not None
                 or selected_slots is not None
                 or bunch_spacing_zeta is not None
                 or int(num_bunches) != 1))

        if slice_mode or bunch_mode:
            # Normalize the public filling inputs into physical filled slots
            # and selected slots. The output bunch axis follows selected_slots.
            if filled_slots is not None and filling_scheme is not None:
                raise ValueError('Only one of `filled_slots` and '
                                 '`filling_scheme` can be provided')
            if filling_scheme is not None:
                filling_scheme = np.asarray(filling_scheme, dtype=np.int64)
                filled_slots = np.nonzero(filling_scheme)[0].astype(np.int64)
            elif filled_slots is not None:
                filled_slots = np.asarray(filled_slots, dtype=np.int64)
            else:
                filled_slots = np.arange(int(num_bunches), dtype=np.int64)

            if len(filled_slots) == 0:
                raise ValueError('At least one filled slot is required')
            if selected_slots is None:
                selected_slots = filled_slots.copy()
            else:
                selected_slots = np.asarray(selected_slots, dtype=np.int64)

            missing = [
                slot for slot in selected_slots if slot not in set(filled_slots)]
            if missing:
                raise ValueError(f'`selected_slots` contains unfilled slots: '
                                 f'{missing}')
            if len(selected_slots) > 1 and bunch_spacing_zeta is None:
                raise ValueError(
                    '`bunch_spacing_zeta` must be provided when more than one '
                    'slot is selected')
            if (bunch_spacing_zeta is None
                    and (len(filled_slots) != 1 or filled_slots[0] != 0
                         or len(selected_slots) != 1
                         or selected_slots[0] != 0)):
                raise ValueError(
                    '`bunch_spacing_zeta` is required unless the monitor uses '
                    'only physical slot 0')
            if (bunch_spacing_zeta is not None
                    and float(bunch_spacing_zeta) <= 0):
                raise ValueError('`bunch_spacing_zeta` must be positive')
        else:
            filled_slots = np.array([], dtype=np.int64)
            selected_slots = np.array([], dtype=np.int64)

        # Dense lookup used by the C kernel to map a physical slot number to
        # the selected-slot axis index in O(1), preserving selected_slots order.
        if len(selected_slots) == 0:
            slot_to_selected = np.array([], dtype=np.int64)
        else:
            max_slot = int(np.max(selected_slots))
            if max_slot < 0:
                raise ValueError('Slot numbers must be non-negative')
            slot_to_selected = np.full(max_slot + 1, -1, dtype=np.int64)
            for ii, slot in enumerate(selected_slots):
                slot = int(slot)
                if slot < 0:
                    raise ValueError('Slot numbers must be non-negative')
                slot_to_selected[slot] = ii

        turns = np.arange(
            int(start_at_turn), int(stop_at_turn), int(every_n_turns),
            dtype=np.int64)
        num_records = len(turns)

        if slice_mode:
            mode = 2
            num_selected_slots = len(selected_slots)
            num_slices_int = int(num_slices)
            zeta_range = tuple(float(vv) for vv in zeta_range)
            z_min_edge = zeta_range[0]
            dzeta = (zeta_range[1] - zeta_range[0]) / num_slices_int
            data_shape = (num_records, num_selected_slots, num_slices_int)
            available_levels = ('beam', 'bunch', 'slice')
            default_level = 'slice'
        elif bunch_mode:
            mode = 1
            num_selected_slots = len(selected_slots)
            num_slices_int = 1
            z_min_edge = (
                0.0 if bunch_spacing_zeta is None
                else -0.5 * float(bunch_spacing_zeta))
            dzeta = 0.0
            data_shape = (num_records, num_selected_slots)
            available_levels = ('beam', 'bunch')
            default_level = 'bunch'
        else:
            mode = 0
            num_selected_slots = 0
            num_slices_int = 0
            z_min_edge = 0.0
            dzeta = 0.0
            data_shape = (num_records,)
            available_levels = ('beam',)
            default_level = 'beam'

        if slice_mode and num_slices_int <= 0:
            raise ValueError('`num_slices` must be positive')
        if slice_mode and dzeta <= 0:
            raise ValueError('`zeta_range` must be increasing')

        flat_size = int(np.prod(data_shape, dtype=np.int64))
        moment_names = ('num_particles', *_moments_for_stats(stats))
        needed_fields = {'num_particles', 'sum_beta0_gamma0'}
        needed_fields.update(_field_name_from_moment(name)
                             for name in moment_names
                             if name != 'num_particles')

        data = {}
        for field in self._RAW_FIELDS:
            size = flat_size if field in needed_fields else 0
            data[field] = size

        super().__init__(
            start_at_turn=int(start_at_turn),
            stop_at_turn=int(stop_at_turn),
            every_n_turns=int(every_n_turns),
            _mode=mode,
            _num_records=num_records,
            _num_selected_slots=num_selected_slots,
            _num_slices=num_slices_int,
            _z_min_edge=z_min_edge,
            _dzeta=dzeta,
            _bunch_spacing_zeta=(
                0.0 if bunch_spacing_zeta is None
                else float(bunch_spacing_zeta)),
            _selected_slots=selected_slots,
            _filled_slots=filled_slots,
            _slot_to_selected=slot_to_selected,
            data=data,
            touched_records={'value': num_records},
            **kwargs)

        self._stats_names = stats
        self._moment_names = moment_names
        self._data_shape = data_shape
        self._available_levels = available_levels
        self._default_level = default_level
        self._output_file = output_file
        if self._output_file is not None:
            self._initialize_output_file()

    @property
    def stats(self):
        """
        Requested public statistic names.
        """
        return self._stats_names

    @property
    def turns(self):
        """
        Machine turns corresponding to the first axis of recorded arrays.
        """
        return np.arange(
            int(self.start_at_turn), int(self.stop_at_turn),
            int(self.every_n_turns), dtype=np.int64)

    @property
    def selected_slots(self):
        """
        Physical bunch slots recorded by the monitor.
        """
        return _to_nparray(self._selected_slots).copy()

    @property
    def filled_slots(self):
        """
        Physical bunch slots considered filled in the monitor configuration.
        """
        return _to_nparray(self._filled_slots).copy()

    @property
    def available_levels(self):
        """
        Aggregation levels available from the recorded primitive moments.
        """
        return self._available_levels

    @property
    def default_level(self):
        """
        Aggregation level returned by statistic attributes and default `get`.
        """
        return self._default_level

    @property
    def zeta_centers(self):
        """
        Longitudinal slice centers for each selected slot, or None.
        """
        if 'slice' not in self.available_levels:
            return None
        base = (float(self._z_min_edge) + (np.arange(int(self._num_slices))
                + 0.5) * float(self._dzeta))
        spacing = float(self._bunch_spacing_zeta)
        return base[None, :] - self.selected_slots[:, None] * spacing

    def __getattr__(self, attr):
        """
        Resolve requested statistic names as computed public attributes.
        """
        if '_stats_names' in self.__dict__ and attr in self._stats_names:
            return self.get(attr)
        return getattr(super(), attr)

    def to_dict(self, **kwargs):
        """
        Return the monitor configuration without logged data.
        """
        out = {
            '__class__': self.__class__.__name__,
            'start_at_turn': int(self.start_at_turn),
            'stop_at_turn': int(self.stop_at_turn),
            'every_n_turns': int(self.every_n_turns),
            'stats': list(self._stats_names),
        }

        if 'slice' in self.available_levels:
            out['zeta_range'] = (
                float(self._z_min_edge),
                float(self._z_min_edge)
                + float(self._dzeta) * int(self._num_slices),
            )
            out['num_slices'] = int(self._num_slices)

        if 'bunch' in self.available_levels:
            out['filled_slots'] = self.filled_slots.tolist()
            out['selected_slots'] = self.selected_slots.tolist()
            if float(self._bunch_spacing_zeta) > 0:
                out['bunch_spacing_zeta'] = float(self._bunch_spacing_zeta)

        return out

    def get(self, stat, *, level=None, turn=None, slot=None, slice_index=None,
            keepdims=False):
        """
        Return a recorded statistic with optional physical selectors.

        Parameters
        ----------
        stat : str
            Requested statistic name.
        level : {"beam", "bunch", "slice"}, optional
            Aggregation level. Defaults to the most detailed available level.
        turn : int or array_like, optional
            Machine turn or turns to select.
        slot : int or array_like, optional
            Physical selected slot or slots to select.
        slice_index : int, optional
            Slice index to select.
        keepdims : bool, optional
            Preserve axes selected by scalar selectors.
        """
        if stat not in self._stats_names:
            raise ValueError(f'Statistic `{stat}` is not recorded')

        # Choose the aggregation level and reject selectors for axes that are
        # not present at that level.
        if level is None:
            level = self.default_level
        elif level not in self.available_levels:
            raise ValueError(
                f'`level` must be one of {self.available_levels}, got '
                f'{level!r}')
        if level == 'beam':
            if slot is not None:
                raise ValueError('`slot` cannot be used with level="beam"')
            if slice_index is not None:
                raise ValueError(
                    '`slice_index` cannot be used with level="beam"')
        elif level == 'bunch' and slice_index is not None:
            raise ValueError(
                '`slice_index` cannot be used with level="bunch"')

        moments = self._moments_at_level(level)
        out = self._compute_stat_from_moments(stat, moments, level=level)

        # Convert physical selectors to array indices. Scalar selectors keep
        # a length-one axis until the final optional squeeze.
        if turn is None:
            turn_selector, turn_is_scalar = slice(None), False
        else:
            turn_selector, turn_is_scalar = _value_indices(
                self.turns, turn, 'turn')
        out = self._apply_selector(out, turn_selector, axis=0)

        slot_is_scalar = False
        slice_is_scalar = False

        if level in ('bunch', 'slice'):
            if slot is None:
                slot_selector, slot_is_scalar = slice(None), False
            else:
                slot_selector, slot_is_scalar = _value_indices(
                    self.selected_slots, slot, 'slot')
            out = self._apply_selector(out, slot_selector, axis=1)

        if level == 'slice':
            if slice_index is None:
                slice_selector, slice_is_scalar = slice(None), False
            else:
                # Accept negative indices with NumPy-like semantics.
                slice_selector = _as_int(slice_index, 'slice_index')
                if slice_selector < 0:
                    slice_selector += int(self._num_slices)
                if (slice_selector < 0
                        or slice_selector >= int(self._num_slices)):
                    raise ValueError(
                        f'`slice_index`={slice_selector} is outside the '
                        'recorded slice range')
                slice_is_scalar = True
            out = self._apply_selector(out, slice_selector, axis=2)

        if not keepdims:
            squeeze_axes = []
            if turn_is_scalar:
                squeeze_axes.append(0)
            if slot_is_scalar and level in ('bunch', 'slice'):
                squeeze_axes.append(1)
            if slice_is_scalar:
                squeeze_axes.append(2)
            for axis in reversed(squeeze_axes):
                out = np.squeeze(out, axis=axis)

        return out

    def record_index(self, turn):
        """
        Return the recorded-array index corresponding to a machine turn.
        """
        return _value_index(self.turns, turn, 'turn')

    def slot_index(self, slot):
        """
        Return the selected-slot axis index for a physical slot number.
        """
        return _value_index(self.selected_slots, slot, 'slot')

    def slice_index(self, zeta, slot=None):
        """
        Return the slice index containing a longitudinal coordinate.
        """
        if 'slice' not in self.available_levels:
            raise ValueError('`zeta` can be mapped only for slice statistics')
        if slot is None:
            if len(self.selected_slots) != 1:
                raise ValueError(
                    '`slot` must be provided when mapping `zeta` with '
                    'multiple selected slots')
            slot = int(self.selected_slots[0])
        self.slot_index(slot)
        zeta_local = float(zeta) + slot * float(self._bunch_spacing_zeta)
        index = int(np.floor((zeta_local - float(self._z_min_edge))
                             / float(self._dzeta)))
        if index < 0 or index >= int(self._num_slices):
            raise ValueError(f'`zeta`={zeta} is outside the monitored '
                             'zeta range')
        return index

    def save_to_file(self):
        """
        Append the current frame to the configured HDF5 output file.

        Only newly touched records are appended. Use :meth:`start_new_frame`
        to clear the in-memory frame and retarget the monitor to later turns.
        """
        if self._output_file is None:
            return

        try:
            import h5py
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ModuleNotFoundError(
                'h5py is required for BeamStatsMonitor HDF5 output'
            ) from exc

        with h5py.File(self._output_file, 'a') as h5file:
            self._initialize_or_validate_hdf5_file(h5file)

            local_start = self._get_local_start_index_from_hdf5(h5file)
            local_stop = self._num_touched_records()
            if local_stop <= local_start:
                return

            record_slice = slice(local_start, local_stop)
            self._append_hdf5_dataset(h5file, 'turns', self.turns[record_slice])

            stats_group = h5file.require_group('stats')
            for level in self.available_levels:
                level_group = stats_group.require_group(level)
                for stat in self.stats:
                    self._append_hdf5_dataset(
                        level_group, stat,
                        self.get(stat, level=level)[record_slice])

            h5file.flush()

    def start_new_frame(self, start_at_turn):
        """
        Clear data and retarget the same-size logged-turn frame.

        The number of records and ``every_n_turns`` are kept fixed. The new
        ``stop_at_turn`` is computed from the existing frame length.
        """
        start_at_turn = _as_int(start_at_turn, 'start_at_turn')
        self.start_at_turn = start_at_turn
        self.stop_at_turn = (
            start_at_turn + int(self._num_records) * int(self.every_n_turns))
        self._reset_data()
        self._reset_touched_records()

    def _moments_at_level(self, level):
        """
        Return primitive moments reduced to the requested aggregation level.
        """
        # Reshape the flat xobjects arrays to the monitor axes before any
        # requested reduction to coarser levels.
        if int(self._mode) == 0:
            data_shape = (int(self._num_records),)
        elif int(self._mode) == 1:
            data_shape = (
                int(self._num_records), int(self._num_selected_slots))
        else:
            data_shape = (
                int(self._num_records), int(self._num_selected_slots),
                int(self._num_slices))

        moments = {}
        for name in self._moment_names:
            if name == 'num_particles':
                field = 'num_particles'
            else:
                field = _field_name_from_moment(name)
            moments[name] = _to_nparray(
                getattr(self.data, field)).reshape(data_shape)
        moments['sum_beta0_gamma0'] = _to_nparray(
            self.data.sum_beta0_gamma0).reshape(data_shape)

        if level == self.default_level:
            return moments

        # Coarser levels are obtained by summing primitive moments, then
        # computing the requested statistic from the reduced moments.
        out = {}
        for name, value in moments.items():
            if self.default_level == 'slice':
                if level == 'bunch':
                    out[name] = np.sum(value, axis=2)
                elif level == 'beam':
                    out[name] = np.sum(value, axis=(1, 2))
            elif self.default_level == 'bunch' and level == 'beam':
                out[name] = np.sum(value, axis=1)
            else:
                out[name] = value
        return out

    def _compute_stat_from_moments(self, name, moments, level):
        """
        Compute one public statistic from primitive moments.
        """
        if name == 'num_particles':
            return moments['num_particles']

        kind, rest = name.split('_', 1)
        if kind == 'mean':
            return self._mean_from_moments(rest, moments)
        if kind == 'sigma':
            var = self._cov_from_moments(rest, rest, moments)
            return np.sqrt(np.maximum(var, 0))
        if kind == 'cov':
            coord1, coord2 = _parse_coord_pair(rest)
            return self._cov_from_moments(coord1, coord2, moments)
        if kind in ('gemitt', 'nemitt'):
            plane = rest.removesuffix('_projected')
            out = self._projected_gemitt_from_moments(plane, moments)
            if kind == 'nemitt':
                weights = moments['num_particles']
                beta0_gamma0 = np.zeros_like(weights, dtype=float)
                np.divide(
                    moments['sum_beta0_gamma0'], weights, out=beta0_gamma0,
                    where=weights > 0)
                out = out * beta0_gamma0
            return out

        raise ValueError(f'Unsupported statistic `{name}`')

    def _mean_from_moments(self, coord, moments):
        """
        Compute a weighted coordinate mean from primitive moments.
        """
        weights = moments['num_particles']
        out = np.zeros_like(weights, dtype=float)
        np.divide(moments[coord], weights, out=out, where=weights > 0)
        return out

    def _cov_from_moments(self, coord1, coord2, moments):
        """
        Compute a weighted covariance from primitive moments.
        """
        weights = moments['num_particles']
        out = np.zeros_like(weights, dtype=float)
        mean_product = (
            self._mean_from_moments(coord1, moments)
            * self._mean_from_moments(coord2, moments))
        np.divide(moments[_moment_name(coord1, coord2)], weights, out=out,
                  where=weights > 0)
        out -= mean_product
        return out

    def _projected_gemitt_from_moments(self, plane, moments):
        """
        Compute projected geometric emittance for one phase-space plane.
        """
        if plane not in _PLANES:
            raise ValueError(f'Unknown projected emittance plane `{plane}`')
        coord, momentum = _PLANES[plane]
        determinant = (
            self._cov_from_moments(coord, coord, moments)
            * self._cov_from_moments(momentum, momentum, moments)
            - self._cov_from_moments(coord, momentum, moments) ** 2)
        return np.sqrt(np.maximum(determinant, 0))

    def _reset_data(self):
        """
        Clear all primitive moment arrays in the current frame.
        """
        for field in self._RAW_FIELDS:
            getattr(self.data, field)[...] = 0.0

    def _reset_touched_records(self):
        """
        Mark all records in the current frame as not touched.
        """
        self.touched_records.value[...] = 0

    def _num_touched_records(self):
        """
        Number of records to write, inferred from touched-record flags.
        """
        touched = np.nonzero(_to_nparray(self.touched_records.value))[0]
        if len(touched) > 0:
            return int(touched[-1]) + 1

        # Fallback for stale prebuilt kernels that do not yet set
        # touched_records. Regenerated/JIT kernels use the explicit flags above.
        weights = self._moments_at_level(self.default_level)['num_particles']
        if weights.ndim == 1:
            nonzero_records = np.nonzero(weights != 0)[0]
        else:
            axes = tuple(range(1, weights.ndim))
            nonzero_records = np.nonzero(np.any(weights != 0, axis=axes))[0]
        if len(nonzero_records) == 0:
            return 0
        return int(nonzero_records[-1]) + 1

    def _get_local_start_index_from_hdf5(self, h5file):
        """
        Return the first current-frame record not already present in HDF5.
        """
        if 'turns' not in h5file or len(h5file['turns']) == 0:
            return 0

        last_turn = int(h5file['turns'][-1])
        turns = self.turns
        if len(turns) == 0:
            return 0

        matches = np.nonzero(turns == last_turn)[0]
        if len(matches) > 0:
            return int(matches[0]) + 1

        if last_turn < turns[0]:
            return 0

        raise RuntimeError(
            'Cannot append BeamStatsMonitor frame because the output file '
            'already contains turns beyond the current frame')

    def _initialize_or_validate_hdf5_file(self, h5file):
        """
        Create or validate the static HDF5 layout and metadata.
        """
        if 'schema_version' not in h5file.attrs:
            if len(h5file.keys()) != 0:
                raise ValueError(
                    'Output HDF5 file is not empty and does not contain '
                    'BeamStatsMonitor metadata')
            self._initialize_hdf5_file(h5file)
        else:
            self._validate_hdf5_file(h5file)

    def _initialize_hdf5_file(self, h5file):
        """
        Initialize static metadata and static datasets.
        """
        h5file.attrs['schema_version'] = 1
        h5file.attrs['class'] = 'BeamStatsMonitor'
        h5file.attrs['stats'] = np.array(self.stats, dtype='S')
        h5file.attrs['available_levels'] = np.array(
            self.available_levels, dtype='S')
        h5file.attrs['default_level'] = self.default_level
        h5file.attrs['every_n_turns'] = int(self.every_n_turns)
        h5file.attrs['n_records_per_frame'] = int(self._num_records)

        h5file.create_dataset(
            'filled_slots', data=self.filled_slots.astype(np.int64))
        h5file.create_dataset(
            'selected_slots', data=self.selected_slots.astype(np.int64))
        if self.zeta_centers is not None:
            h5file.create_dataset('zeta_centers', data=self.zeta_centers)

    def _initialize_output_file(self):
        """
        Create a fresh HDF5 output file for this monitor.
        """
        try:
            import h5py
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ModuleNotFoundError(
                'h5py is required for BeamStatsMonitor HDF5 output'
            ) from exc

        with h5py.File(self._output_file, 'w') as h5file:
            self._initialize_hdf5_file(h5file)
            h5file.flush()

    def _validate_hdf5_file(self, h5file):
        """
        Validate high-level compatibility with an existing HDF5 file.
        """
        expected_attrs = {
            'schema_version': 1,
            'class': 'BeamStatsMonitor',
            'stats': np.array(self.stats, dtype='S'),
            'available_levels': np.array(self.available_levels, dtype='S'),
            'default_level': self.default_level,
            'every_n_turns': int(self.every_n_turns),
            'n_records_per_frame': int(self._num_records),
        }
        for name, expected in expected_attrs.items():
            if name not in h5file.attrs:
                raise ValueError(
                    f'Output HDF5 file is missing metadata `{name}`')
            actual = h5file.attrs[name]
            if isinstance(expected, np.ndarray):
                if not np.array_equal(np.asarray(actual), expected):
                    raise ValueError(
                        f'Output HDF5 metadata `{name}` does not match '
                        'this monitor')
            elif actual != expected:
                raise ValueError(
                    f'Output HDF5 metadata `{name}` does not match this '
                    'monitor')

        self._check_hdf5_dataset_equal(
            h5file, 'filled_slots', self.filled_slots.astype(np.int64))
        self._check_hdf5_dataset_equal(
            h5file, 'selected_slots', self.selected_slots.astype(np.int64))

    @staticmethod
    def _check_hdf5_dataset_equal(group, name, expected):
        if name not in group:
            raise ValueError(f'Output HDF5 file is missing dataset `{name}`')
        if not np.array_equal(group[name][...], expected):
            raise ValueError(
                f'Output HDF5 dataset `{name}` does not match this monitor')

    @staticmethod
    def _append_hdf5_dataset(group, name, data):
        """
        Append an array along the first axis of an HDF5 dataset.
        """
        data = np.asarray(data)
        tail_shape = data.shape[1:]
        if name in group:
            dataset = group[name]
            if dataset.shape[1:] != tail_shape:
                raise ValueError(
                    f'Output HDF5 dataset `{dataset.name}` has shape '
                    f'{dataset.shape}, expected tail shape {tail_shape}')
        else:
            dataset = group.create_dataset(
                name,
                shape=(0, *tail_shape),
                maxshape=(None, *tail_shape),
                chunks=(1, *tail_shape),
                dtype=data.dtype)
        old_size = dataset.shape[0]
        new_size = old_size + data.shape[0]
        dataset.resize((new_size, *dataset.shape[1:]))
        dataset[old_size:new_size] = data

    @staticmethod
    def _apply_selector(array, selector, axis):
        """
        Apply a scalar/list/slice selector to one array axis.
        """
        if isinstance(selector, slice):
            indices = [slice(None)] * array.ndim
            indices[axis] = selector
            return array[tuple(indices)]
        return np.take(array, np.atleast_1d(selector), axis=axis)


def _value_index(values, value, name):
    """
    Return the index of a scalar value in a one-dimensional array.
    """
    value = _as_int(value, name)
    matches = np.nonzero(values == value)[0]
    if len(matches) == 0:
        raise ValueError(f'`{name}`={value} is not recorded')
    return int(matches[0])


def _value_indices(values, value, name):
    """
    Return indices for scalar or array-like physical selector values.
    """
    value_array = np.asarray(value)
    is_scalar = value_array.ndim == 0
    flat_values = value_array.reshape(-1)
    indices = np.array([
        _value_index(values, item, name) for item in flat_values],
        dtype=np.int64)
    return indices, is_scalar


def _as_int(value, name):
    """
    Convert a value to int while rejecting non-integral values.
    """
    value_as_int = int(value)
    if value_as_int != value:
        raise ValueError(f'`{name}` must be an integer')
    return value_as_int


def _check_supported_stats(stats):
    """
    Validate requested public statistics and raise for unsupported names.
    """
    unsupported_coupled = [
        name for name in stats
        if (name.startswith('gemitt_') or name.startswith('nemitt_'))
        and not name.endswith('_projected')
    ]
    if unsupported_coupled:
        raise NotImplementedError(
            'Coupled normal-mode emittances are not implemented yet. '
            'Use `gemitt_*_projected` or `nemitt_*_projected` for now.')

    for name in stats:
        if name == 'num_particles':
            continue
        if name.startswith('mean_'):
            _check_coord(name[5:])
        elif name.startswith('sigma_'):
            _check_coord(name[6:])
        elif name.startswith('cov_'):
            _parse_coord_pair(name[4:])
        elif name.startswith('gemitt_') or name.startswith('nemitt_'):
            plane = name.split('_', 1)[1].removesuffix('_projected')
            if plane not in _PLANES:
                raise ValueError(f'Unknown emittance plane `{plane}`')
        else:
            raise ValueError(f'Unsupported statistic `{name}`')


def _moments_for_stats(stats):
    """
    Return primitive moment names required by public statistics.
    """
    moments = set()
    for name in stats:
        if name == 'num_particles':
            continue
        if name.startswith('mean_'):
            moments.add(name[5:])
        elif name.startswith('sigma_'):
            coord = name[6:]
            moments.add(coord)
            moments.add(_moment_name(coord, coord))
        elif name.startswith('cov_'):
            coord1, coord2 = _parse_coord_pair(name[4:])
            moments.add(coord1)
            moments.add(coord2)
            moments.add(_moment_name(coord1, coord2))
        elif name.startswith('gemitt_') or name.startswith('nemitt_'):
            plane = name.split('_', 1)[1].removesuffix('_projected')
            coord, momentum = _PLANES[plane]
            moments.update([
                coord, momentum,
                _moment_name(coord, coord),
                _moment_name(momentum, momentum),
                _moment_name(coord, momentum),
            ])
    return tuple(sorted(moments))


def _check_coord(coord):
    """
    Validate a particle coordinate name.
    """
    if coord not in _COORDS:
        raise ValueError(f'Unknown coordinate `{coord}`')


def _parse_coord_pair(name):
    """
    Parse and validate a coordinate-pair name.
    """
    parts = name.split('_')
    if len(parts) != 2:
        raise ValueError(f'Invalid coordinate pair `{name}`')
    coord1, coord2 = parts
    _check_coord(coord1)
    _check_coord(coord2)
    return coord1, coord2


def _moment_name(coord1, coord2):
    """
    Return the canonical primitive second-moment name for two coordinates.
    """
    _check_coord(coord1)
    _check_coord(coord2)
    i1 = _COORDS.index(coord1)
    i2 = _COORDS.index(coord2)
    if i1 <= i2:
        return f'{coord1}_{coord2}'
    return f'{coord2}_{coord1}'


def _field_name_from_moment(name):
    """
    Return the record field name used to store a primitive moment.
    """
    if name in _COORDS or name in _SECOND_MOMENTS:
        return f'sum_{name}'
    raise ValueError(f'Unknown moment `{name}`')


def _to_nparray(array):
    """
    Convert context or xobjects arrays to a NumPy array.
    """
    if hasattr(array, 'to_nparray'):
        return array.to_nparray()
    if hasattr(array, 'get'):
        return array.get()
    return np.asarray(array)
