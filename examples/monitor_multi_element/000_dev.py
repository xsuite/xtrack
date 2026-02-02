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


