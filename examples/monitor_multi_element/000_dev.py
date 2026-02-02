import xtrack as xt
import xobjects as xo
from pathlib import Path


# TODO:
# - Add new monitor to pre-compiled kernels
# - Forbid backtrack for now
# - Forbid collective mode for now
# - Forbid GPU for now

import numpy as np

line = xt.load('../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')

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

multi_elem_monit = xt.MultiElementMonitor(
    start_at_turn=start_at_turn,
    stop_at_turn=stop_at_turn,
    part_id_start=part_id_start,
    part_id_end=part_id_end,
    at_element_mapping=at_element_mapping,
    data=(n_turns, n_particles, n_coordinates, n_elements)
)

p = xt.Particles(p0c=7e12, x=np.arange(20)*1e-12,
                           delta=np.arange(20)*1e-12,
                           at_element=4027,
                           at_turn=5
)

multi_elem_monit.track(p)

multi_elem_monit.data[:,:,5,172]  # delta at BPM with index 172

line._multi_element_monitor = multi_elem_monit

p = xt.Particles(p0c=7e12, x=1e-6*np.arange(20),
                           delta=0
)
line.track(p, num_turns=10)