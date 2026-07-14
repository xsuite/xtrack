import numpy as np

from ..beam_elements import Marker
from ..slicers import ElementWithSlicer


_COORDS = ('x', 'px', 'y', 'py', 'zeta', 'delta')
_PLANES = {
    'x': ('x', 'px'),
    'y': ('y', 'py'),
    'zeta': ('zeta', 'delta'),
}

_DEFAULT_STATS = (
    'num_particles',
    'mean_x', 'mean_y',
    'sigma_x', 'sigma_y',
)


class BeamStatsMonitor(ElementWithSlicer):
    """
    Monitor weighted beam statistics on a longitudinal slicing grid.

    The stored `num_particles` is the sum of particle weights in each bin.
    All derived quantities are weighted by the same particle weights.

    Parameters
    ----------
    start_at_turn : int, optional
        First turn to record, inclusive.
    stop_at_turn : int, optional
        Last turn to record, exclusive. If not provided, only
        `start_at_turn` is recorded.
    every_n_turns : int, optional
        Record one turn every `every_n_turns` turns.
    zeta_range : tuple[float, float]
        Longitudinal range covered by the slicer. For bunched beams this is
        the range around each selected bunch. For coasting beams this is the
        full monitored longitudinal range.
    num_slices : int
        Number of longitudinal slices per selected bunch or coasting domain.
    coasting : bool, optional
        If ``True``, treat the beam as one longitudinal domain and hide the
        artificial bunch axis in public accessors by default.
    num_bunches : int, optional
        Number of consecutive filled slots, starting from slot 0. Used only
        when neither `filled_slots` nor `filling_scheme` is provided.
    filling_scheme : array_like of int or bool, optional
        Low-level filling scheme. Non-zero entries identify filled physical
        slots.
    filled_slots : array_like of int, optional
        Physical slot numbers which are filled. This is an alternative to
        `filling_scheme`.
    selected_slots : array_like of int, optional
        Physical slot numbers to record. If omitted, all filled slots are
        recorded. The output bunch axis follows this order.
    bunch_spacing_zeta : float, optional
        Longitudinal spacing between adjacent physical slots.
    stats : sequence of str, optional
        Statistics to record. Supported values are ``"num_particles"``,
        ``"mean_<coord>"``, ``"sigma_<coord>"``, ``"cov_<coord1>_<coord2>"``,
        ``"gemitt_<plane>_projected"``, and
        ``"nemitt_<plane>_projected"``, where coordinates are ``x``, ``px``,
        ``y``, ``py``, ``zeta``, and ``delta``, and planes are ``x``, ``y``,
        and ``zeta``.
    output_file : str or path-like, optional
        Reserved for future HDF5 output support.
    storage : str, optional
        Reserved for future storage mode selection. Only ``None`` and
        ``"memory"`` are currently accepted.
    buffer_size : int, optional
        Reserved for future HDF5 buffering support.
    **kwargs
        Additional keyword arguments passed to the underlying xobjects
        initialization.

    Notes
    -----
    Public statistic arrays have shape
    ``(n_logged_turns, n_selected_slots, num_slices)`` for bunched beams. In
    coasting mode the selected-slot axis is hidden by default, giving shape
    ``(n_logged_turns, num_slices)``.

    Coupled normal-mode emittances are intentionally not enabled yet. Request
    projected emittances with the ``_projected`` suffix.
    """

    allow_loss_refinement = True

    def __init__(self, *,
                 start_at_turn=0,
                 stop_at_turn=None,
                 every_n_turns=1,
                 zeta_range=None,
                 num_slices=None,
                 coasting=False,
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

        if zeta_range is None:
            raise ValueError('`zeta_range` must be provided')
        if num_slices is None:
            raise ValueError('`num_slices` must be provided')
        if stop_at_turn is None:
            stop_at_turn = start_at_turn + 1
        if every_n_turns <= 0:
            raise ValueError('`every_n_turns` must be positive')

        stats = _DEFAULT_STATS if stats is None else tuple(stats)
        stats = _normalize_stats(stats)
        _check_supported_stats(stats)

        (filled_slots, selected_slots, bunch_selection,
         filling_scheme) = _normalize_filling(
             num_bunches=num_bunches,
             filled_slots=filled_slots,
             filling_scheme=filling_scheme,
             selected_slots=selected_slots,
             coasting=coasting)

        self.start_at_turn = int(start_at_turn)
        self.stop_at_turn = int(stop_at_turn)
        self.every_n_turns = int(every_n_turns)
        self._turns = np.arange(
            self.start_at_turn, self.stop_at_turn, self.every_n_turns,
            dtype=np.int64)
        self._coasting = bool(coasting)
        self._stats_names = stats
        self._selected_slots = selected_slots.copy()
        self._filled_slots = filled_slots.copy()
        self._bunch_selection = bunch_selection.copy()

        slicer_moments = _moments_for_stats(stats)
        super().__init__(
            slicer_moments=slicer_moments,
            zeta_range=zeta_range,
            num_slices=num_slices,
            bunch_spacing_zeta=bunch_spacing_zeta,
            filling_scheme=filling_scheme,
            filled_slots=filled_slots,
            bunch_selection=bunch_selection,
            **kwargs)

        data_shape = (len(self._turns), len(selected_slots), int(num_slices))
        self._data = {}
        for name in stats:
            fill_value = 0.0 if name == 'num_particles' else np.nan
            self._data[name] = np.full(data_shape, fill_value, dtype=float)

    @property
    def stats(self):
        """
        Recorded statistic names.

        Returns
        -------
        tuple of str
            Names of the statistics recorded by this monitor.
        """
        return self._stats_names

    @property
    def turns(self):
        """
        Logged turn numbers.

        Returns
        -------
        numpy.ndarray
            One-dimensional array containing the machine turns stored in the
            first axis of each recorded statistic.
        """
        return self._turns.copy()

    @property
    def selected_slots(self):
        """
        Physical slot numbers recorded by the monitor.

        Returns
        -------
        numpy.ndarray
            One-dimensional array of selected physical slot numbers. The order
            matches the bunch axis of stored bunched-beam data.
        """
        return self._selected_slots.copy()

    @property
    def filled_slots(self):
        """
        Physical slot numbers present in the beam.

        Returns
        -------
        numpy.ndarray
            One-dimensional array of filled physical slot numbers.
        """
        return self._filled_slots.copy()

    @property
    def coasting(self):
        """
        Whether the monitor uses coasting-beam output conventions.

        Returns
        -------
        bool
            ``True`` if the artificial bunch axis is hidden by default in
            public accessors.
        """
        return self._coasting

    @property
    def zeta_centers(self):
        """
        Longitudinal slice centers.

        Returns
        -------
        numpy.ndarray
            Slice-center coordinates. For bunched beams the shape is
            ``(n_selected_slots, num_slices)``. In coasting mode the shape is
            ``(num_slices,)``.
        """
        return self._public_shape(self._as_np_2d(self.slicer.zeta_centers))

    def __getattr__(self, attr):
        if '_data' in self.__dict__ and attr in self._data:
            return self.get(attr)
        return getattr(super(), attr)

    def get(self, stat, keep_bunch_axis=False):
        """
        Return one recorded statistic.

        Parameters
        ----------
        stat : str
            Name of the recorded statistic to return.
        keep_bunch_axis : bool, optional
            If ``True``, always return the canonical shape
            ``(n_logged_turns, n_selected_slots, num_slices)``. If ``False``,
            coasting-mode output drops the artificial selected-slot axis.

        Returns
        -------
        numpy.ndarray
            Recorded data for the requested statistic.

        Raises
        ------
        ValueError
            If `stat` was not requested when constructing the monitor.
        """
        if stat not in self._data:
            raise ValueError(f'Statistic `{stat}` is not recorded')
        return self._public_shape(self._data[stat], keep_bunch_axis)

    def track(self, particles, _slice_result=None, _other_bunch_slicers=None):
        """
        Record beam statistics for the current turn when selected.

        Parameters
        ----------
        particles : xtrack.Particles
            Particles to slice and accumulate.
        _slice_result : dict, optional
            Internal precomputed slicing result shared by collective elements.
        _other_bunch_slicers : sequence, optional
            Internal slicer data received from other pipeline partners.

        Returns
        -------
        None or xtrack.PipelineStatus
            ``None`` when tracking can continue immediately. A
            ``PipelineStatus`` is returned when pipeline communication puts the
            element on hold.

        Notes
        -----
        The current turn is read from ``particles.at_turn``. Turns outside
        ``[start_at_turn, stop_at_turn)`` or not aligned with
        `every_n_turns` are skipped without slicing.
        """
        turn = self._current_turn(particles)
        if not self._logs_turn(turn):
            return None

        status = super().track(
            particles,
            _slice_result=_slice_result,
            _other_bunch_slicers=_other_bunch_slicers)
        if status is not None:
            return status

        i_record = (turn - self.start_at_turn) // self.every_n_turns
        self._record(i_record, particles)
        return None

    def get_backtrack_element(self, _context=None, _buffer=None, _offset=None):
        return Marker(_context=_context, _buffer=_buffer, _offset=_offset)

    def _current_turn(self, particles):
        return int(self._context.nparray_from_context_array(
            particles.at_turn[:1])[0])

    def _logs_turn(self, turn):
        if turn < self.start_at_turn or turn >= self.stop_at_turn:
            return False
        return (turn - self.start_at_turn) % self.every_n_turns == 0

    def _record(self, i_record, particles):
        for name in self._stats_names:
            self._data[name][i_record, :, :] = self._compute_stat(
                name, particles)

    def _compute_stat(self, name, particles):
        if name == 'num_particles':
            return self._weights()

        kind, rest = name.split('_', 1)
        if kind == 'mean':
            return self._mean(rest)
        if kind == 'sigma':
            return self._sigma(rest)
        if kind == 'cov':
            coord1, coord2 = _parse_coord_pair(rest)
            return self._cov(coord1, coord2)
        if kind in ('gemitt', 'nemitt'):
            plane = rest.removesuffix('_projected')
            out = self._projected_gemitt(plane)
            if kind == 'nemitt':
                out = out * self._beta0_gamma0(particles)
            return out

        raise ValueError(f'Unsupported statistic `{name}`')

    def _weights(self):
        return self._as_np_2d(self.slicer.num_particles)

    def _sum(self, coord1, coord2=None):
        if coord2 is None:
            return self._as_np_2d(self.slicer.sum(coord1))
        return self._as_np_2d(self.slicer.sum(coord1, coord2))

    def _mean(self, coord):
        weights = self._weights()
        out = np.full_like(weights, np.nan, dtype=float)
        np.divide(self._sum(coord), weights, out=out, where=weights > 0)
        return out

    def _cov(self, coord1, coord2):
        weights = self._weights()
        out = np.full_like(weights, np.nan, dtype=float)
        mean_product = self._mean(coord1) * self._mean(coord2)
        np.divide(self._sum(coord1, coord2), weights, out=out,
                  where=weights > 0)
        out -= mean_product
        return out

    def _sigma(self, coord):
        var = self._cov(coord, coord)
        return np.sqrt(np.maximum(var, 0))

    def _projected_gemitt(self, plane):
        if plane not in _PLANES:
            raise ValueError(f'Unknown projected emittance plane `{plane}`')
        coord, momentum = _PLANES[plane]
        determinant = (
            self._cov(coord, coord) * self._cov(momentum, momentum)
            - self._cov(coord, momentum) ** 2)
        return np.sqrt(np.maximum(determinant, 0))

    def _beta0_gamma0(self, particles):
        beta0 = self._context.nparray_from_context_array(
            particles.beta0[:1])[0]
        gamma0 = self._context.nparray_from_context_array(
            particles.gamma0[:1])[0]
        return float(beta0 * gamma0)

    def _as_np_2d(self, array):
        out = np.asarray(self._context.nparray_from_context_array(array))
        if out.ndim == 1:
            out = out.reshape(1, -1)
        return out

    def _public_shape(self, array, keep_bunch_axis=False):
        if self.coasting and not keep_bunch_axis:
            return array[:, 0, :] if array.ndim == 3 else array[0, :]
        return array


def _normalize_stats(stats):
    out = []
    for stat in stats:
        if stat not in out:
            out.append(stat)
    return tuple(out)


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
            moments.add(f'{coord}_{coord}')
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
    return sorted(moments)


def _normalize_filling(*, num_bunches, filled_slots, filling_scheme,
                       selected_slots, coasting):
    if coasting:
        if filling_scheme is not None or filled_slots is not None:
            raise ValueError(
                '`filling_scheme` and `filled_slots` are not used in '
                'coasting mode')
        filled_slots = np.array([0], dtype=np.int64)
        selected_slots = np.array([0], dtype=np.int64)
        bunch_selection = np.array([0], dtype=np.int64)
        return filled_slots, selected_slots, bunch_selection, None

    if filled_slots is not None and filling_scheme is not None:
        raise ValueError(
            'Only one of `filled_slots` and `filling_scheme` can be provided')

    if filling_scheme is not None:
        filling_scheme = np.asarray(filling_scheme, dtype=np.int64)
        filled_slots = np.nonzero(filling_scheme)[0].astype(np.int64)
    elif filled_slots is not None:
        filled_slots = np.asarray(filled_slots, dtype=np.int64)
    else:
        filled_slots = np.arange(int(num_bunches), dtype=np.int64)

    if selected_slots is None:
        selected_slots = filled_slots.copy()
    else:
        selected_slots = np.asarray(selected_slots, dtype=np.int64)

    slot_to_index = {slot: ii for ii, slot in enumerate(filled_slots)}
    try:
        bunch_selection = np.array(
            [slot_to_index[slot] for slot in selected_slots],
            dtype=np.int64)
    except KeyError as err:
        raise ValueError(
            f'Selected slot {err.args[0]} is not in `filled_slots`') from None

    if len(selected_slots) == 0:
        raise ValueError('At least one slot must be selected')

    return filled_slots, selected_slots, bunch_selection, filling_scheme


def _check_coord(coord):
    if coord not in _COORDS:
        raise ValueError(f'Unknown coordinate `{coord}`')


def _parse_coord_pair(name):
    for coord1 in _COORDS:
        prefix = coord1 + '_'
        if name.startswith(prefix):
            coord2 = name[len(prefix):]
            _check_coord(coord2)
            return coord1, coord2
    raise ValueError(f'Cannot parse coordinate pair `{name}`')


def _moment_name(coord1, coord2):
    _check_coord(coord1)
    _check_coord(coord2)
    index1 = _COORDS.index(coord1)
    index2 = _COORDS.index(coord2)
    if index1 <= index2:
        return f'{coord1}_{coord2}'
    return f'{coord2}_{coord1}'
