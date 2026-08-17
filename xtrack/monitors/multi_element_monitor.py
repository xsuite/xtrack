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
                 **kwargs):
        super().__init__(start_at_turn=start_at_turn,
                         stop_at_turn=stop_at_turn,
                         part_id_start=part_id_start,
                         part_id_end=part_id_end,
                         at_element_mapping=at_element_mapping,
                         data=data,
                         map_slots=map_slots,
                         **kwargs)
        self.obs_names = obs_names
        self._name_to_index = {
            name: idx for idx, name in enumerate(self.obs_names)
        }
        self._map_series = None  # filled by a ParticlesTpsa track
        self._map_ref_particle = None

    def __len__(self):
        return len(self.obs_names)

    def _obs_index(self, obs_name):
        if isinstance(obs_name, str):
            try:
                return self._name_to_index[obs_name]
            except KeyError:
                raise KeyError(f'{obs_name!r} is not a recorded location') from None
        return obs_name

    def _recorded_maps(self, turn):
        if self._map_series is None:
            raise AttributeError(
                'No TPSA maps recorded, this monitor only holds doubles')
        return self._map_series[turn]

    def map_at(self, obs_name, turn=0):
        """The full TPSA map recorded at a location, sharing the series."""
        from xtrack.tpsa.particles import ParticlesTpsa

        return ParticlesTpsa._from_coords(
            self._recorded_maps(turn)[self._obs_index(obs_name)],
            self._map_ref_particle)

    def map_jacobian(self, turn=0):
        """Recorded transfer matrices, one per location, `(num_locations, 6, 6)`."""
        return np.array([[series.grad() for series in location]
                         for location in self._recorded_maps(turn)])

    def __repr__(self):
        obs_names_print = (self.obs_names if len(self.obs_names) < 5
                           else list(self.obs_names[:5]) + ['...'])
        obs_names_str = ', '.join(obs_names_print)
        return (f'MultiElementMonitor('
                f'obs_names=[{obs_names_str}])')

    @staticmethod
    def build_map_slots(descriptor, num_locations, num_turns):
        """Preallocated series for the full TPSA maps, and the addresses C writes to.

        ``data`` holds doubles, so a map passing through leaves only its constant part
        behind. These series take the whole polynomial instead (``mad_tpsa_copy`` in the
        C loop), so one pass yields a complete map at every location.

        Returns ``(series, addresses)`` with ``series[turn][location][coordinate]``. The
        series have to outlive the track, since the C writes through their addresses.
        """
        import xgtpsa
        from xtrack.tpsa.particles import _COORDS

        series = [
            [[descriptor.zero() for _ in _COORDS] for _ in range(num_locations)]
            for _ in range(num_turns)
        ]
        ffi = xgtpsa.ffi()
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
            turn_index = turn - self.start_at_turn
        else:
            turn_index = slice(None)

        return self.data[turn_index, particle_index, coord_index, obs_index]
