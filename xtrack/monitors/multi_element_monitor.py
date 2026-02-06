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
                 **kwargs):
        super().__init__(start_at_turn=start_at_turn,
                         stop_at_turn=stop_at_turn,
                         part_id_start=part_id_start,
                         part_id_end=part_id_end,
                         at_element_mapping=at_element_mapping,
                         data=data,
                         **kwargs)
        self.obs_names = obs_names
        self._name_to_index = {
            name: idx for idx, name in enumerate(self.obs_names)
        }

    def __repr__(self):
        obs_names_print = (self.obs_names if len(self.obs_names) < 5
                           else list(self.obs_names[:5]) + ['...'])
        obs_names_str = ', '.join(obs_names_print)
        return (f'MultiElementMonitor('
                f'obs_names=[{obs_names_str}])')

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
