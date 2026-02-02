import xtrack as xt
import xobjects as xo

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
mask_bpm = tt.rows.mask.match(name='bpm.*')
bpm_names = tt.name[mask_bpm]