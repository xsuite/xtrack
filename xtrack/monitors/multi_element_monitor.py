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