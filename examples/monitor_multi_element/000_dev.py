import xtrack as xt
import xobjects as xo

import numpy as np

line = xt.load('../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')

class MultiElementMonitor(xt.BeamElement):
    _xofields = {
        'start_at_turn': xo.Int64,
        'stop_at_turn': xo.Int64,
        'part_id_start': xo.Int64,
        'part_id_end': xo.Int64,
        'at_element_mapping': xo.Int64[:],
        'data': xo.Float64[:, :, :, :], # turns, particles, coordinate, location
    }

tt = line.get_table()
indeces_bpm = tt.rows.indices.match(name='bpm.*')
obs_names = tt.name[indeces_bpm]

at_element_mapping = np.zeros(len(tt), dtype=np.int64)
at_element_mapping[:] = -1
at_element_mapping[indeces_bpm] = np.arange(len(indeces_bpm), dtype=np.int64)

start_at_turn = 0
stop_at_turn = 10
part_id_start = 5
part_id_end = 12

n_turns = stop_at_turn - start_at_turn
n_particles = part_id_end - part_id_start
n_elements = len(indeces_bpm)
n_coordinates = 6

multi_elem_monit = MultiElementMonitor(
    start_at_turn=start_at_turn,
    stop_at_turn=stop_at_turn,
    part_id_start=part_id_start,
    part_id_end=part_id_end,
    at_element_mapping=at_element_mapping,
    data=(n_turns, n_particles, n_coordinates, n_elements)
)
