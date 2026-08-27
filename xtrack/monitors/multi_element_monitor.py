import numpy as np

import xtrack as xt
import xobjects as xo


class MultiElementMonitor(xt.BeamElement):
    _xofields = {
        'start_at_turn': xo.Int64,
        'stop_at_turn': xo.Int64,
        'part_id_start': xo.Int64,
        'part_id_end': xo.Int64,
        'at_element_mapping': xo.Int64[:],
        'data': xo.Float64[:, :, :, :], # turns, particles, coordinate, location
        # Destinations for the full TPSA maps, empty in scalar tracking
        'map_slots': xo.UInt64[:, :, :], # turns, location, coordinate
        # Selected map coefficients instead of full map
        'monomial_indices': xo.Int64[:],  # descriptor coefficient index per slot
        'coord_indices': xo.Int64[:],     # output coordinate index per slot
        'coefficients': xo.Float64[:, :, :], # turns, location, slot
    }

    behaves_like_drift = True
    has_backtrack = True
    allow_loss_refinement = True

    _extra_c_sources = [
        '#include "xtrack/monitors/multi_element_monitor.h"',
    ]

    _coord_name_to_index = {'x': 0, 'px': 1, 'y': 2, 'py': 3,
                            'zeta': 4, 'delta': 5, 's': 6}

    def __init__(self, start_at_turn, stop_at_turn,
                 part_id_start, part_id_end,
                 at_element_mapping,
                 data,
                 obs_names,
                 map_slots=(0, 0, 0),
                 monomial_slots=None,
                 **kwargs):
        num_slots = 0 if monomial_slots is None else len(monomial_slots)
        num_turns = stop_at_turn - start_at_turn
        super().__init__(start_at_turn=start_at_turn,
                         stop_at_turn=stop_at_turn,
                         part_id_start=part_id_start,
                         part_id_end=part_id_end,
                         at_element_mapping=at_element_mapping,
                         data=data,
                         map_slots=map_slots,
                         monomial_indices=num_slots,
                         coord_indices=num_slots,
                         coefficients=(num_turns, len(obs_names), num_slots),
                         **kwargs)
        self.obs_names = obs_names
        self._name_to_index = {
            name: idx for idx, name in enumerate(self.obs_names)
        }
        self._map_series = None  # filled by a ParticlesTpsa track
        self._map_ref_particle = None
        self.monomial_slots = None
        if monomial_slots is not None:
            self.monomial_slots = [(m, c) for m, c, _ in monomial_slots]
            self._fill_monomial_slots(monomial_slots)

    def __len__(self):
        return len(self.obs_names)

    def _obs_index(self, obs_name):
        if isinstance(obs_name, str):
            try:
                return self._name_to_index[obs_name]
            except KeyError:
                raise KeyError(f'{obs_name!r} is not a recorded location') from None
        return obs_name

    def _turn_index(self, turn):
        """Array index of an absolute turn number."""
        if not self.start_at_turn <= turn < self.stop_at_turn:
            raise IndexError(
                f'turn {turn} not recorded, this monitor covers turns '
                f'[{self.start_at_turn}, {self.stop_at_turn})')
        return turn - self.start_at_turn

    def _fill_monomial_slots(self, monomial_slots):
        """Write the C lookup arrays and the slot index, from `parse_monomials`."""
        self._slot_index = {}
        for slot, (monomial, coord, coefficient_index) in enumerate(monomial_slots):
            self.monomial_indices[slot] = coefficient_index
            self.coord_indices[slot] = self._coord_name_to_index[coord]
            self._slot_index[monomial, coord] = slot

    @staticmethod
    def parse_monomials(monomials, descriptor):
        """Slots `(monomial, coord, coefficient_index)` for one track.

        `monomials` is an `(N, 6+np)` array of monomials, recorded for all six
        coordinates, or a mapping `{monomial: coord}` / `{monomial: (coord, ...)}`.
        """
        from xtrack.tpsa.particles import _COORDS

        if hasattr(monomials, 'items'):
            requested = list(monomials.items())
        else:
            requested = [(monomial, _COORDS) for monomial in np.asarray(monomials)]

        slots = []
        seen = set()
        for monomial, coords in requested:
            monomial = tuple(int(order) for order in np.asarray(monomial).reshape(-1))
            if len(monomial) != descriptor.monomial_length:
                raise ValueError(
                    f'invalid monomial {monomial}: expected length '
                    f'{descriptor.monomial_length} (6 vars + '
                    f'{descriptor.num_params} params)')
            if not descriptor.is_valid_monomial(monomial):
                raise ValueError(
                    f'invalid monomial {monomial}: beyond the order or the '
                    f'parameter order of the descriptor')
            coefficient_index = descriptor.monomial_index(monomial)
            if isinstance(coords, str):
                coords = (coords,)
            for coord in coords:
                if coord not in _COORDS:
                    raise ValueError(
                        f'{coord!r} is not an output coordinate, expected one '
                        f'of {list(_COORDS)}')
                if (monomial, coord) in seen:
                    raise ValueError(
                        f'monomial {monomial} requested twice for {coord!r}')
                seen.add((monomial, coord))
                slots.append((monomial, coord, coefficient_index))
        if not slots:
            raise ValueError('no monomials to record')
        return slots

    def coefficient(self, monomial, coord=None, obs_name=None, turn=None):
        """Recorded coefficient(s) of `monomial`, axes `(turns, locations, coords)`.

        An axis is dropped when selected. `turn` is an absolute turn number.
        """
        if self.monomial_slots is None:
            raise AttributeError(
                'No coefficients recorded, this monitor holds full TPSA maps')

        monomial = tuple(int(order) for order in np.asarray(monomial).reshape(-1))
        if coord is None:
            coords = [c for m, c in self.monomial_slots if m == monomial]
            if not coords:
                raise KeyError(f'monomial {monomial} was not recorded')
            slots = [self._slot_index[monomial, c] for c in coords]
        else:
            if not isinstance(coord, str):
                coord = list(self._coord_name_to_index)[coord]
            try:
                slots = self._slot_index[monomial, coord]
            except KeyError:
                raise KeyError(
                    f'monomial {monomial} was not recorded for {coord!r}') from None

        turn_index = slice(None) if turn is None else self._turn_index(turn)
        obs_index = slice(None) if obs_name is None else self._obs_index(obs_name)
        return np.asarray(self.coefficients)[turn_index, obs_index, slots]

    def _recorded_maps(self, turn=None):
        if self._map_series is None:
            raise AttributeError(
                'No TPSA maps recorded, this monitor only holds doubles')
        if turn is None:
            turn = self.start_at_turn
        return self._map_series[self._turn_index(turn)]

    def map_at(self, obs_name, turn=None):
        """The full TPSA map recorded at a location, sharing the series."""
        from xtrack.tpsa.particles import ParticlesTpsa

        return ParticlesTpsa._from_coords(
            self._recorded_maps(turn)[self._obs_index(obs_name)],
            self._map_ref_particle)

    def map_jacobian(self, turn=None):
        """Recorded transfer matrices, one per location, `(num_locations, 6, 6)`."""
        return np.array([[series.grad() for series in location]
                         for location in self._recorded_maps(turn)])

    def __repr__(self):
        obs_names_print = (self.obs_names if len(self.obs_names) < 5
                           else list(self.obs_names[:5]) + ['...'])
        obs_names_str = ', '.join(obs_names_print)
        if self.monomial_slots is None:
            recorded = ''
        else:
            recorded = f', monomial_slots={len(self.monomial_slots)}'
        return (f'MultiElementMonitor('
                f'obs_names=[{obs_names_str}]{recorded})')

    @staticmethod
    def build_map_slots(descriptor, num_locations, num_turns):
        """Preallocated series for the full TPSA maps, and the addresses C writes to.

        ``data`` holds doubles, so a map passing through leaves only its constant part
        behind. These series take the whole polynomial instead (``mad_tpsa_copy`` in the
        C loop), so one pass yields a complete map at every location.

        Returns ``(series, addresses)`` with ``series[turn][location][coordinate]``. The
        series have to outlive the track, since the C writes through their addresses.
        """
        import madng_tpsa
        from xtrack.tpsa.particles import _COORDS

        series = [
            [[descriptor.zero() for _ in _COORDS] for _ in range(num_locations)]
            for _ in range(num_turns)
        ]
        ffi = madng_tpsa.ffi()
        addresses = np.array(
            [
                int(ffi.cast("uintptr_t", one_series.ptr))
                for turn in series
                for location in turn
                for one_series in location
            ],
            dtype=np.uint64,
        ).reshape(num_turns, num_locations, len(_COORDS))
        return series, addresses

    def get(self, coordinate, obs_name=None, particle_id=None, turn=None):
        coord_index = self._coord_name_to_index[coordinate]

        if obs_name is None:
            obs_index = slice(None)
        else:
            obs_index = self._name_to_index[obs_name]

        if particle_id is not None:
            particle_index = particle_id - self.part_id_start
        else:
            particle_index = slice(None)

        if turn is not None:
            turn_index = self._turn_index(turn)
        else:
            turn_index = slice(None)

        return self.data[turn_index, particle_index, coord_index, obs_index]
