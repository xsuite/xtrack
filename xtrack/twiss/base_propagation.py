# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from .open_twiss import _twiss_open


def _propagate_twiss_from_init(
        line, init, start, end, nemitt_x, nemitt_y, step_W_sigma,
        delta_disp, use_full_inverse, hide_thin_groups, only_markers,
        only_orbit, spin, compute_lattice_functions, continue_if_lost,
        keep_tracking_data, keep_initial_particles, initial_particles,
        ebe_monitor):

    return _twiss_open(
        line=line,
        init=init,
        start=start, end=end,
        nemitt_x=nemitt_x,
        nemitt_y=nemitt_y,
        step_W_sigma=step_W_sigma,
        delta_disp=delta_disp,
        use_full_inverse=use_full_inverse,
        hide_thin_groups=hide_thin_groups,
        only_markers=only_markers,
        only_orbit=only_orbit,
        spin=spin,
        compute_lattice_functions=compute_lattice_functions,
        _continue_if_lost=continue_if_lost,
        _keep_tracking_data=keep_tracking_data,
        _keep_initial_particles=keep_initial_particles,
        _initial_particles=initial_particles,
        _ebe_monitor=ebe_monitor)
