import numpy as np
import xobjects as xo

from ..base_element import BeamElement
from ..beam_elements import Marker


_COORDS = ('x', 'px', 'y', 'py', 'zeta', 'delta')
_PLANES = {
    'x': ('x', 'px'),
    'y': ('y', 'py'),
    'zeta': ('zeta', 'delta'),
}

_SECOND_MOMENTS = tuple(
    f'{coord1}_{coord2}'
    for ii, coord1 in enumerate(_COORDS)
    for coord2 in _COORDS[ii:]
)

_DEFAULT_STATS = (
    'num_particles',
    'mean_x', 'mean_y',
    'sigma_x', 'sigma_y',
)


class BeamStatsMonitorRecord(xo.Struct):
    num_particles = xo.Float64[:]
    sum_beta0_gamma0 = xo.Float64[:]
    sum_x = xo.Float64[:]
    sum_px = xo.Float64[:]
    sum_y = xo.Float64[:]
    sum_py = xo.Float64[:]
    sum_zeta = xo.Float64[:]
    sum_delta = xo.Float64[:]
    sum_x_x = xo.Float64[:]
    sum_x_px = xo.Float64[:]
    sum_x_y = xo.Float64[:]
    sum_x_py = xo.Float64[:]
    sum_x_zeta = xo.Float64[:]
    sum_x_delta = xo.Float64[:]
    sum_px_px = xo.Float64[:]
    sum_px_y = xo.Float64[:]
    sum_px_py = xo.Float64[:]
    sum_px_zeta = xo.Float64[:]
    sum_px_delta = xo.Float64[:]
    sum_y_y = xo.Float64[:]
    sum_y_py = xo.Float64[:]
    sum_y_zeta = xo.Float64[:]
    sum_y_delta = xo.Float64[:]
    sum_py_py = xo.Float64[:]
    sum_py_zeta = xo.Float64[:]
    sum_py_delta = xo.Float64[:]
    sum_zeta_zeta = xo.Float64[:]
    sum_zeta_delta = xo.Float64[:]
    sum_delta_delta = xo.Float64[:]


class BeamStatsMonitor(BeamElement):
    """
    Monitor weighted beam statistics.

    This monitor stores primitive weighted sums directly in xobjects arrays.
    Derived statistics are computed on access from those primitive sums.
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
        'data': BeamStatsMonitorRecord,
    }

    _extra_c_sources = [
        '#include "xtrack/monitors/beam_stats_monitor.h"',
    ]

    behaves_like_drift = True
    has_backtrack = True
    allow_loss_refinement = True
    allow_no_prebuilt_kernel = True

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
                 storage=None,
                 buffer_size=None,
                 _xobject=None,
                 **kwargs):

        if _xobject is not None:
            super().__init__(_xobject=_xobject)
            return

        if output_file is not None or storage not in (None, 'memory'):
            raise NotImplementedError(
                'BeamStatsMonitor HDF5/file storage is not implemented yet')
        if buffer_size is not None:
            raise NotImplementedError(
                '`buffer_size` is reserved for HDF5/file storage')

        slice_mode = zeta_range is not None or num_slices is not None
        if (zeta_range is None) != (num_slices is None):
            raise ValueError(
                '`zeta_range` and `num_slices` must be provided together')
        if stop_at_turn is None:
            stop_at_turn = start_at_turn + 1
        if every_n_turns <= 0:
            raise ValueError('`every_n_turns` must be positive')

        stats = _DEFAULT_STATS if stats is None else tuple(stats)
        stats = _normalize_stats(stats)
        _check_supported_stats(stats)

        bunch_mode = (not slice_mode and _has_bunch_inputs(
            num_bunches=num_bunches,
            filled_slots=filled_slots,
            filling_scheme=filling_scheme,
            selected_slots=selected_slots,
            bunch_spacing_zeta=bunch_spacing_zeta))

        if slice_mode or bunch_mode:
            filled_slots, selected_slots = _normalize_filling(
                num_bunches=num_bunches,
                filled_slots=filled_slots,
                filling_scheme=filling_scheme,
                selected_slots=selected_slots)
            if len(selected_slots) > 1 and bunch_spacing_zeta is None:
                raise ValueError(
                    '`bunch_spacing_zeta` must be provided when more than one '
                    'slot is selected')
        else:
            filled_slots = np.array([], dtype=np.int64)
            selected_slots = np.array([], dtype=np.int64)

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
            if bunch_spacing_zeta is None:
                z_min_edge = 0.0
                dzeta = 0.0
            else:
                half_spacing = 0.5 * float(bunch_spacing_zeta)
                z_min_edge = -half_spacing
                dzeta = 2.0 * half_spacing
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
            data[field] = np.zeros(size, dtype=float)

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
            data=data,
            **kwargs)

        self._stats_names = stats
        self._moment_names = moment_names
        self._data_shape = data_shape
        self._available_levels = available_levels
        self._default_level = default_level

    @property
    def stats(self):
        return self._stats_names

    @property
    def turns(self):
        return np.arange(
            int(self.start_at_turn), int(self.stop_at_turn),
            int(self.every_n_turns), dtype=np.int64)

    @property
    def selected_slots(self):
        return _to_nparray(self._selected_slots).copy()

    @property
    def filled_slots(self):
        return _to_nparray(self._filled_slots).copy()

    @property
    def available_levels(self):
        return self._available_levels

    @property
    def default_level(self):
        return self._default_level

    @property
    def zeta_centers(self):
        if 'slice' not in self.available_levels:
            return None
        base = (float(self._z_min_edge) + (np.arange(int(self._num_slices))
                + 0.5) * float(self._dzeta))
        spacing = float(self._bunch_spacing_zeta)
        return base[None, :] - self.selected_slots[:, None] * spacing

    def __getattr__(self, attr):
        if '_stats_names' in self.__dict__ and attr in self._stats_names:
            return self.get(attr)
        return getattr(super(), attr)

    def get(self, stat, *, level=None, turn=None, slot=None, slice_index=None,
            zeta=None, keepdims=False):
        if stat not in self._stats_names:
            raise ValueError(f'Statistic `{stat}` is not recorded')

        level = self._normalize_level(level)
        self._check_selectors_for_level(
            level=level, slot=slot, slice_index=slice_index, zeta=zeta)

        if slice_index is not None and zeta is not None:
            raise ValueError('Only one of `slice_index` and `zeta` can be '
                             'provided')
        if zeta is not None:
            slice_index = self.slice_index(zeta, slot=slot)

        moments = self._moments_at_level(level)
        out = self._compute_stat_from_moments(stat, moments, level=level)

        turn_selector, turn_is_scalar = self._turn_selector(turn)
        out = self._apply_selector(out, turn_selector, axis=0)

        slot_is_scalar = False
        slice_is_scalar = False

        if level in ('bunch', 'slice'):
            slot_selector, slot_is_scalar = self._slot_selector(slot)
            out = self._apply_selector(out, slot_selector, axis=1)

        if level == 'slice':
            slice_selector, slice_is_scalar = self._slice_selector(
                slice_index)
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
        return _value_index(self.turns, turn, 'turn')

    def slot_index(self, slot):
        return _value_index(self.selected_slots, slot, 'slot')

    def slice_index(self, zeta, slot=None):
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

    def get_backtrack_element(self, _context=None, _buffer=None, _offset=None):
        return Marker(_context=_context, _buffer=_buffer, _offset=_offset)

    def _shape(self):
        if int(self._mode) == 0:
            return (int(self._num_records),)
        if int(self._mode) == 1:
            return (int(self._num_records), int(self._num_selected_slots))
        return (int(self._num_records), int(self._num_selected_slots),
                int(self._num_slices))

    def _moment_array(self, name):
        if name == 'num_particles':
            field = 'num_particles'
        else:
            field = _field_name_from_moment(name)
        arr = _to_nparray(getattr(self.data, field))
        return arr.reshape(self._shape())

    def _stored_moments(self):
        out = {name: self._moment_array(name) for name in self._moment_names}
        out['sum_beta0_gamma0'] = _to_nparray(
            self.data.sum_beta0_gamma0).reshape(self._shape())
        return out

    def _moments_at_level(self, level):
        moments = self._stored_moments()
        if level == self.default_level:
            return moments

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
        if name == 'num_particles':
            return moments['num_particles']

        kind, rest = name.split('_', 1)
        if kind == 'mean':
            return self._mean_from_moments(rest, moments)
        if kind == 'sigma':
            return self._sigma_from_moments(rest, moments)
        if kind == 'cov':
            coord1, coord2 = _parse_coord_pair(rest)
            return self._cov_from_moments(coord1, coord2, moments)
        if kind in ('gemitt', 'nemitt'):
            plane = rest.removesuffix('_projected')
            out = self._projected_gemitt_from_moments(plane, moments)
            if kind == 'nemitt':
                out = out * self._beta0_gamma0_from_moments(moments)
            return out

        raise ValueError(f'Unsupported statistic `{name}`')

    def _mean_from_moments(self, coord, moments):
        weights = moments['num_particles']
        out = np.full_like(weights, np.nan, dtype=float)
        np.divide(moments[coord], weights, out=out, where=weights > 0)
        return out

    def _cov_from_moments(self, coord1, coord2, moments):
        weights = moments['num_particles']
        out = np.full_like(weights, np.nan, dtype=float)
        mean_product = (
            self._mean_from_moments(coord1, moments)
            * self._mean_from_moments(coord2, moments))
        np.divide(moments[_moment_name(coord1, coord2)], weights, out=out,
                  where=weights > 0)
        out -= mean_product
        return out

    def _sigma_from_moments(self, coord, moments):
        var = self._cov_from_moments(coord, coord, moments)
        return np.sqrt(np.maximum(var, 0))

    def _projected_gemitt_from_moments(self, plane, moments):
        if plane not in _PLANES:
            raise ValueError(f'Unknown projected emittance plane `{plane}`')
        coord, momentum = _PLANES[plane]
        determinant = (
            self._cov_from_moments(coord, coord, moments)
            * self._cov_from_moments(momentum, momentum, moments)
            - self._cov_from_moments(coord, momentum, moments) ** 2)
        return np.sqrt(np.maximum(determinant, 0))

    def _beta0_gamma0_from_moments(self, moments):
        weights = moments['num_particles']
        out = np.full_like(weights, np.nan, dtype=float)
        np.divide(moments['sum_beta0_gamma0'], weights, out=out,
                  where=weights > 0)
        return out

    def _normalize_level(self, level):
        if level is None:
            return self.default_level
        if level not in self.available_levels:
            raise ValueError(
                f'`level` must be one of {self.available_levels}, got '
                f'{level!r}')
        return level

    def _check_selectors_for_level(self, *, level, slot, slice_index, zeta):
        if level == 'beam':
            if slot is not None:
                raise ValueError('`slot` cannot be used with level="beam"')
            if slice_index is not None or zeta is not None:
                raise ValueError(
                    '`slice_index` and `zeta` cannot be used with '
                    'level="beam"')
        elif level == 'bunch':
            if slice_index is not None or zeta is not None:
                raise ValueError(
                    '`slice_index` and `zeta` cannot be used with '
                    'level="bunch"')

    def _turn_selector(self, turn):
        if turn is None:
            return slice(None), False
        return _value_indices(self.turns, turn, 'turn')

    def _slot_selector(self, slot):
        if slot is None:
            return slice(None), False
        return _value_indices(self.selected_slots, slot, 'slot')

    def _slice_selector(self, slice_index):
        if slice_index is None:
            return slice(None), False
        return _normalize_slice_index(
            slice_index, int(self._num_slices), 'slice_index'), True

    @staticmethod
    def _apply_selector(array, selector, axis):
        if isinstance(selector, slice):
            indices = [slice(None)] * array.ndim
            indices[axis] = selector
            return array[tuple(indices)]
        return np.take(array, np.atleast_1d(selector), axis=axis)


def _normalize_stats(stats):
    out = []
    for stat in stats:
        if stat not in out:
            out.append(stat)
    return tuple(out)


def _has_bunch_inputs(*, num_bunches, filled_slots, filling_scheme,
                      selected_slots, bunch_spacing_zeta):
    return (
        filled_slots is not None
        or filling_scheme is not None
        or selected_slots is not None
        or bunch_spacing_zeta is not None
        or int(num_bunches) != 1)


def _normalize_filling(*, num_bunches, filled_slots, filling_scheme,
                       selected_slots):
    if filled_slots is not None and filling_scheme is not None:
        raise ValueError('Only one of `filled_slots` and `filling_scheme` can '
                         'be provided')
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

    missing = [slot for slot in selected_slots if slot not in set(filled_slots)]
    if missing:
        raise ValueError(f'`selected_slots` contains unfilled slots: '
                         f'{missing}')
    return filled_slots, selected_slots


def _value_index(values, value, name):
    value = _as_int(value, name)
    matches = np.nonzero(values == value)[0]
    if len(matches) == 0:
        raise ValueError(f'`{name}`={value} is not recorded')
    return int(matches[0])


def _value_indices(values, value, name):
    value_array = np.asarray(value)
    is_scalar = value_array.ndim == 0
    flat_values = value_array.reshape(-1)
    indices = np.array([
        _value_index(values, item, name) for item in flat_values],
        dtype=np.int64)
    return indices, is_scalar


def _normalize_slice_index(index, num_slices, name):
    index = _as_int(index, name)
    if index < 0:
        index += num_slices
    if index < 0 or index >= num_slices:
        raise ValueError(f'`{name}`={index} is outside the recorded slice '
                         'range')
    return index


def _as_int(value, name):
    value_as_int = int(value)
    if value_as_int != value:
        raise ValueError(f'`{name}` must be an integer')
    return value_as_int


def _check_supported_stats(stats):
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
    if coord not in _COORDS:
        raise ValueError(f'Unknown coordinate `{coord}`')


def _parse_coord_pair(name):
    parts = name.split('_')
    if len(parts) != 2:
        raise ValueError(f'Invalid coordinate pair `{name}`')
    coord1, coord2 = parts
    _check_coord(coord1)
    _check_coord(coord2)
    return coord1, coord2


def _moment_name(coord1, coord2):
    _check_coord(coord1)
    _check_coord(coord2)
    i1 = _COORDS.index(coord1)
    i2 = _COORDS.index(coord2)
    if i1 <= i2:
        return f'{coord1}_{coord2}'
    return f'{coord2}_{coord1}'


def _field_name_from_moment(name):
    if name in _COORDS or name in _SECOND_MOMENTS:
        return f'sum_{name}'
    raise ValueError(f'Unknown moment `{name}`')


def _to_nparray(array):
    if hasattr(array, 'to_nparray'):
        return array.to_nparray()
    if hasattr(array, 'get'):
        return array.get()
    return np.asarray(array)
