# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np
import xobjects as xo

from .lattice_functions_from_W import _get_lattice_functions
from .twiss_table import TwissTable

import xtrack as xt  # To avoid circular imports


AT_TURN_FOR_TWISS = -10  # Avoid writing in monitors installed in the line.


def _propagate_twiss_from_init(
        line,
        init,
        start,
        end,
        nemitt_x,
        nemitt_y,
        step_W_sigma,
        delta_disp,
        use_full_inverse,
        hide_thin_groups=False,
        only_markers=False,
        only_orbit=False,
        spin=False,
        compute_lattice_functions=True,
        continue_if_lost=False,
        keep_tracking_data=False,
        keep_initial_particles=False,
        initial_particles=None,
        ebe_monitor=None,
):
    """Track an orbit and basis particles from a completed Twiss init."""

    if init.reference_frame == 'reverse':
        init = init.reverse()

    particle_on_co = init.particle_on_co
    W_matrix = init.W_matrix

    if start is not None and end is None:
        raise ValueError('end must be specified if start is not None')

    if end is not None and start is None:
        raise ValueError('start must be specified if end is not None')

    if start is None:
        start = 0

    if isinstance(start, str):
        start = line._element_names_unique.index(start)
    if isinstance(end, str):
        if end == '_end_point':
            end = len(line._element_names_unique) - 1
        else:
            end = line._element_names_unique.index(end)

    if init.element_name == line._element_names_unique[start]:
        twiss_orientation = 'forward'
    elif init.element_name == '_end_point' and end == len(line._element_names_unique) - 1:
        twiss_orientation = 'backward'
    elif end is not None and init.element_name == line._element_names_unique[end]:
        twiss_orientation = 'backward'
    else:
        raise ValueError(
            '``init`` must be given at the start or end of the specified element range.')

    ctx2np = line._context.nparray_from_context_array

    gemitt_x = nemitt_x/particle_on_co._xobject.beta0[0]/particle_on_co._xobject.gamma0[0]
    gemitt_y = nemitt_y/particle_on_co._xobject.beta0[0]/particle_on_co._xobject.gamma0[0]
    scale_transverse_x = np.sqrt(gemitt_x)*step_W_sigma
    scale_transverse_y = np.sqrt(gemitt_y)*step_W_sigma
    scale_longitudinal = delta_disp
    scale_eigen = min(scale_transverse_x, scale_transverse_y, scale_longitudinal)

    context = line._context
    if initial_particles is not None: # used in match
        part_for_twiss = initial_particles.copy()
    else:
        import xpart
        part_for_twiss = xpart.build_particles(_context=context,
            particle_ref=particle_on_co, mode='shift',
            include_collective=True,
            x     = [0] + list(W_matrix[0, :] * -scale_eigen) + list(W_matrix[0, :] * scale_eigen),
            px    = [0] + list(W_matrix[1, :] * -scale_eigen) + list(W_matrix[1, :] * scale_eigen),
            y     = [0] + list(W_matrix[2, :] * -scale_eigen) + list(W_matrix[2, :] * scale_eigen),
            py    = [0] + list(W_matrix[3, :] * -scale_eigen) + list(W_matrix[3, :] * scale_eigen),
            zeta  = [0] + list(W_matrix[4, :] * -scale_eigen) + list(W_matrix[4, :] * scale_eigen),
            pzeta = [0] + list(W_matrix[5, :] * -scale_eigen) + list(W_matrix[5, :] * scale_eigen),
            )
        part_for_twiss.ax = particle_on_co._xobject.ax[0]
        part_for_twiss.ay = particle_on_co._xobject.ay[0]
        if spin:
            part_for_twiss.spin_x = particle_on_co._xobject.spin_x[0]
            part_for_twiss.spin_y = particle_on_co._xobject.spin_y[0]
            part_for_twiss.spin_z = particle_on_co._xobject.spin_z[0]

        if twiss_orientation == 'forward':
            part_for_twiss.at_element = start
            part_for_twiss.s = line.tracker._tracker_data_base.element_s_locations[start]
        elif twiss_orientation == 'backward':
            part_for_twiss.at_element = end + 1 # to include the last element
            part_for_twiss.s = line.tracker._tracker_data_base.element_s_locations[end]
        else:
            raise ValueError('Invalid twiss_orientation')

    part_for_twiss.at_turn = AT_TURN_FOR_TWISS # To avoid writing in monitors

    if keep_initial_particles:
        part_for_twiss0 = part_for_twiss.copy()

    if ebe_monitor is not None:
        _monitor = ebe_monitor
    elif hasattr(line.tracker._tracker_data_base, '_reusable_ebe_monitor_for_twiss'):
        _monitor = line.tracker._tracker_data_base._reusable_ebe_monitor_for_twiss
    else:
        _monitor = 'ONE_TURN_EBE'

    if end is None:
        ele_stop_track = None
    else:
        ele_stop_track = end + 1 # to include the last element

    with xt.line._preserve_config(line):
        if spin:
            # Spin is behind the same compile flag as synchrotron radiation
            line.config.XTRACK_MULTIPOLE_NO_SYNRAD = False
        line.track(part_for_twiss, turn_by_turn_monitor=_monitor,
                    ele_start=start,
                    ele_stop=ele_stop_track,
                    backtrack=(twiss_orientation == 'backward'))

    # We keep the monitor to speed up future calls (attached to tracker data
    # so that it is trashed if number of elements changes)
    line.tracker._tracker_data_base._reusable_ebe_monitor_for_twiss = line.record_last_track

    if not continue_if_lost:
        assert np.all(ctx2np(part_for_twiss.state) == 1), (
            'Some test particles were lost during twiss! '
          + f'(state {np.unique(ctx2np(part_for_twiss.state))}, '
          + f'at element {np.unique(ctx2np(part_for_twiss.at_element))})')

    if twiss_orientation == 'forward':
        i_start = start
        i_stop = part_for_twiss._xobject.at_element[0] + (
                (part_for_twiss._xobject.at_turn[0] - AT_TURN_FOR_TWISS)
                * len(line._element_names_unique))
    elif twiss_orientation == 'backward':
        i_start = start
        if ele_stop_track is not None:
            i_stop = ele_stop_track
        else:
            i_stop = len(line._element_names_unique) - 1

    recorded_state = line.record_last_track.state[:, i_start:i_stop+1].copy()
    if not continue_if_lost:
        assert np.all(recorded_state == 1), (
             'Some test particles were lost during twiss! '
          + f'(state {np.unique(recorded_state)}, '
          + f'at element {np.unique(line.record_last_track.at_element[:, i_start:i_stop+1].copy())})')

    x_co = line.record_last_track.x[0, i_start:i_stop+1].copy()
    y_co = line.record_last_track.y[0, i_start:i_stop+1].copy()
    px_co = line.record_last_track.px[0, i_start:i_stop+1].copy()
    py_co = line.record_last_track.py[0, i_start:i_stop+1].copy()
    zeta_co = line.record_last_track.zeta[0, i_start:i_stop+1].copy()
    delta_co = np.array(line.record_last_track.delta[0, i_start:i_stop+1].copy())
    ptau_co = np.array(line.record_last_track.ptau[0, i_start:i_stop+1].copy())
    s_co = line.record_last_track.s[0, i_start:i_stop+1].copy()
    kin_px_co = line.record_last_track.kin_px[0, i_start:i_stop+1].copy()
    kin_py_co = line.record_last_track.kin_py[0, i_start:i_stop+1].copy()
    kin_ps_co = line.record_last_track.kin_ps[0, i_start:i_stop+1].copy()
    kin_xp_co = line.record_last_track.kin_xp[0, i_start:i_stop+1].copy()
    kin_yp_co = line.record_last_track.kin_yp[0, i_start:i_stop+1].copy()
    if spin:
        spin_x_co = line.record_last_track.spin_x[0, i_start:i_stop+1].copy()
        spin_y_co = line.record_last_track.spin_y[0, i_start:i_stop+1].copy()
        spin_z_co = line.record_last_track.spin_z[0, i_start:i_stop+1].copy()

    Ws = np.zeros(shape=(len(s_co), 6, 6), dtype=np.float64)
    Ws[:, 0, :] = 0.5 * (line.record_last_track.x[1:7, i_start:i_stop+1] - x_co).T / scale_eigen
    Ws[:, 1, :] = 0.5 * (line.record_last_track.px[1:7, i_start:i_stop+1] - px_co).T / scale_eigen
    Ws[:, 2, :] = 0.5 * (line.record_last_track.y[1:7, i_start:i_stop+1] - y_co).T / scale_eigen
    Ws[:, 3, :] = 0.5 * (line.record_last_track.py[1:7, i_start:i_stop+1] - py_co).T / scale_eigen
    Ws[:, 4, :] = 0.5 * (line.record_last_track.zeta[1:7, i_start:i_stop+1] - zeta_co).T / scale_eigen
    Ws[:, 5, :] = 0.5 * (line.record_last_track.ptau[1:7, i_start:i_stop+1] - ptau_co).T / particle_on_co._xobject.beta0[0] / scale_eigen

    Ws[:, 0, :] -= 0.5 * (line.record_last_track.x[7:13, i_start:i_stop+1] - x_co).T / scale_eigen
    Ws[:, 1, :] -= 0.5 * (line.record_last_track.px[7:13, i_start:i_stop+1] - px_co).T / scale_eigen
    Ws[:, 2, :] -= 0.5 * (line.record_last_track.y[7:13, i_start:i_stop+1] - y_co).T / scale_eigen
    Ws[:, 3, :] -= 0.5 * (line.record_last_track.py[7:13, i_start:i_stop+1] - py_co).T / scale_eigen
    Ws[:, 4, :] -= 0.5 * (line.record_last_track.zeta[7:13, i_start:i_stop+1] - zeta_co).T / scale_eigen
    Ws[:, 5, :] -= 0.5 * (line.record_last_track.ptau[7:13, i_start:i_stop+1] - ptau_co).T / particle_on_co._xobject.beta0[0] / scale_eigen

    name_co = np.array(line._element_names_unique[i_start:i_stop] + ('_end_point',))
    name_co_env = np.array(line.element_names[i_start:i_stop] + ('_end_point',))

    if only_markers:
        raise NotImplementedError('only_markers not supported anymore')

    twiss_res_element_by_element = {}

    twiss_res_element_by_element.update({
        'name': name_co,
        's': s_co,
        'x': x_co,
        'px': px_co,
        'y': y_co,
        'py': py_co,
        'zeta': zeta_co,
        'delta': delta_co,
        'ptau': ptau_co,
        'W_matrix': Ws,
        'kin_px': kin_px_co,
        'kin_py': kin_py_co,
        'kin_ps': kin_ps_co,
        'kin_xp': kin_xp_co,
        'kin_yp': kin_yp_co,
        'kin_xprime': kin_xp_co,
        'kin_yprime': kin_yp_co,
        'env_name': name_co_env,
    })
    if spin:
        twiss_res_element_by_element.update({
            'spin_x': spin_x_co,
            'spin_y': spin_y_co,
            'spin_z': spin_z_co,
        })

    if not only_orbit and compute_lattice_functions:
        lattice_functions, i_replace = _get_lattice_functions(Ws, use_full_inverse, s_co)
        twiss_res_element_by_element.update(lattice_functions)

    extra_data = {}
    extra_data['only_markers'] = only_markers
    if keep_tracking_data:
        extra_data['tracking_data'] = line.record_last_track.copy()

    if keep_initial_particles:
        extra_data['_initial_particles'] = part_for_twiss0.copy()

    if hide_thin_groups:
        _vars_hide_changes = [
            'x', 'px', 'y', 'py', 'zeta', 'delta', 'ptau',
            'betx', 'bety', 'alfx', 'alfy', 'gamx', 'gamy',
            'betx1', 'bety1', 'betx2', 'bety2',
            'betx_edw_teng', 'bety_edw_teng',
            'alfx_edw_teng', 'alfy_edw_teng', 'g_edw_teng',
            'f1001', 'f1010', 'f0110', 'f0101',
            'dx', 'dpx', 'dy', 'dpy',
        ]

        for key in _vars_hide_changes:
            if key in twiss_res_element_by_element:
                twiss_res_element_by_element[key][i_replace] = np.nan

    twiss_res_element_by_element['name'] = np.array(twiss_res_element_by_element['name'])

    twiss_res = TwissTable(data=twiss_res_element_by_element)
    twiss_res._data.update(extra_data)

    twiss_res._data['particle_on_co'] = particle_on_co.copy(_context=xo.context_default)

    line_length = line.tracker._tracker_data_base.line_length
    twiss_res._data['line_length'] = line_length
    twiss_res._data['circumference'] = line_length # deprecated
    twiss_res._data['_orientation'] = twiss_orientation

    return twiss_res
