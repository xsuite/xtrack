from __future__ import annotations

from typing import Sequence

import madng_tpsa
import numpy as np

import xtrack as xt
import xobjects as xo
from xtrack.tpsa.particles import COORDS, ParticlesTpsa

# A monomial is the per-variable orders, a tuple once parsed. A request is
# several of them, or the ones wanted per output coordinate.
Monomial = Sequence[int]
MonomialRequest = (list[Monomial] | np.ndarray
                   | dict[str, Monomial | list[Monomial]])


def _parse_monomials(monomials: MonomialRequest,
                     descriptor: madng_tpsa.Descriptor) -> list[tuple]:
    """Expand the requested monomials into a list of tuples.

    Each requested coefficient becomes one recording `(coord, monomial,
    coefficient_index)`, where `coefficient_index` describes the flat index of the
    coefficient in the descriptor. C reads it with `mad_tpsa_geti`.
    `monomials` is several monomials, recorded for all six output coordinates,
    or a `{coord: monomial}` / `{coord: monomials}` mapping.
    """
    if isinstance(monomials, dict):
        requested = list(monomials.items())
    else:
        requested = [(coord, monomials) for coord in COORDS]

    # list of (coord, monomial, coefficient_index) tuples
    recordings = []
    seen = set()
    for coord, coord_monomials in requested:
        if coord not in COORDS:
            raise ValueError(
                f'{coord!r} is not an output coordinate, expected one '
                f'of {list(COORDS)}')
        rows = np.asarray(coord_monomials)
        rows = rows.reshape(1, -1) if rows.ndim == 1 else rows
        for row in rows:
            monomial = tuple(int(order) for order in row)
            if len(monomial) != descriptor.monomial_length:
                raise ValueError(
                    f'Invalid monomial {monomial}: expected length '
                    f'{descriptor.monomial_length} (6 vars + '
                    f'{descriptor.num_params} params)')
            if not descriptor.is_valid_monomial(monomial):
                raise ValueError(
                    f'Invalid monomial {monomial}: beyond the order or the '
                    f'parameter order of the descriptor')
            if (coord, monomial) in seen:
                raise ValueError(
                    f'Monomial {monomial} requested twice for {coord!r}')
            seen.add((coord, monomial))
            recordings.append(
                (coord, monomial, descriptor.monomial_index(monomial)))
    if not recordings:
        raise ValueError('No monomials to record')
    return recordings


class MultiElementMonitor(xt.BeamElement):
    _xofields = {
        'start_at_turn': xo.Int64,
        'stop_at_turn': xo.Int64,
        'part_id_start': xo.Int64,
        'part_id_end': xo.Int64,
        'at_element_mapping': xo.Int64[:],
        'data': xo.Float64[:, :, :, :], # turns, particles, coordinate, location
        # Addresses of the tpsa_t the full maps are copied into, empty in
        # scalar tracking. Kept alive by `_map_series`.
        'tpsa_addresses': xo.UInt64[:, :, :], # turns, location, coordinate
        # Selected map coefficients instead of full maps, one slot per
        # recorded (coord, monomial) pair.
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
                 tpsa_addresses=(0, 0, 0),
                 monomials: MonomialRequest | None = None,
                 descriptor: madng_tpsa.Descriptor | None = None,
                 **kwargs):
        """Records coordinates at the observed locations, as well as
        the TPSA maps when tracking TPSAs.

        `tpsa_addresses` are where the full maps are copied, see
        `build_tpsa_addresses`. Give `monomials` and the TPSA `descriptor`
        instead to record only selected map coefficients, read back with
        `coefficient`. The two are exclusive, `monomials` allocates no series.
        """

        recordings = ([] if monomials is None
                      else _parse_monomials(monomials, descriptor))
        num_turns = stop_at_turn - start_at_turn
        super().__init__(start_at_turn=start_at_turn,
                         stop_at_turn=stop_at_turn,
                         part_id_start=part_id_start,
                         part_id_end=part_id_end,
                         at_element_mapping=at_element_mapping,
                         data=data,
                         tpsa_addresses=tpsa_addresses,
                         monomial_indices=len(recordings),
                         coord_indices=len(recordings),
                         coefficients=(num_turns, len(obs_names), len(recordings)),
                         **kwargs)
        self.obs_names = obs_names
        self._name_to_index = {
            name: idx for idx, name in enumerate(self.obs_names)
        }
        self._map_series = None  # filled by a ParticlesTpsa track
        self._map_ref_particle = None

        # `(coord, monomial)` per slot, and the reverse lookup
        self.recorded_monomials = None if monomials is None else [
            (coord, monomial) for coord, monomial, _ in recordings]
        self._slot_index = {}
        for slot, (coord, monomial, coefficient_index) in enumerate(recordings):
            self.monomial_indices[slot] = coefficient_index
            self.coord_indices[slot] = self._coord_name_to_index[coord]
            self._slot_index[coord, monomial] = slot

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

    def coefficient(self, coord: str | int, monomial: Monomial,
                    obs_name: str | int | None = None, turn: int | None = None):
        """Recorded coefficient of `monomial` in the `coord` series.

        Axes `(turns, locations)`, each dropped when `turn` or `obs_name` selects
        one. `turn` is an absolute turn number.
        """
        if self.recorded_monomials is None:
            raise AttributeError(
                'No coefficients recorded, this monitor holds full TPSA maps')
        if not isinstance(coord, str):
            coord = COORDS[coord]
        monomial = tuple(int(order) for order in monomial)
        try:
            slot = self._slot_index[coord, monomial]
        except KeyError:
            raise KeyError(f'Monomial {monomial} was not recorded '
                           f'for {coord!r}') from None

        turn_index = slice(None) if turn is None else self._turn_index(turn)
        obs_index = slice(None) if obs_name is None else self._obs_index(obs_name)
        return np.asarray(self.coefficients)[turn_index, obs_index, slot]

    def _recorded_maps(self, turn=None):
        if self._map_series is None:
            raise AttributeError(
                'No TPSA maps recorded, this monitor only holds doubles')
        if turn is None:
            turn = self.start_at_turn
        return self._map_series[self._turn_index(turn)]

    def map_at(self, obs_name, turn=None):
        """The full TPSA map recorded at a location, sharing the series."""
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
        if self.recorded_monomials is None:
            recorded = ''
        else:
            recorded = f', monomials={len(self.recorded_monomials)}'
        return (f'MultiElementMonitor('
                f'obs_names=[{obs_names_str}]{recorded})')

    @staticmethod
    def build_tpsa_addresses(descriptor: madng_tpsa.Descriptor,
                             num_locations: int, num_turns: int):
        """Preallocated series for the full TPSA maps, and the addresses C writes to.

        ``data`` holds doubles, so a map passing through leaves only its constant part
        behind. These series take the whole polynomial instead (``mad_tpsa_copy`` in the
        C loop), so one pass yields a complete map at every location.

        Returns ``(series, addresses)`` with ``series[turn][location][coordinate]``.
        ``addresses`` are raw ``tpsa_t*``, so the series have to outlive the track.
        """
        series = [
            [[descriptor.zero() for _ in COORDS] for _ in range(num_locations)]
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
        ).reshape(num_turns, num_locations, len(COORDS))
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
