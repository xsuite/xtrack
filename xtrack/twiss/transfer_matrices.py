# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from warnings import warn

import numpy as np

from ..general import DEPRECATION_INFO_PREP_1_0
from .constants import AT_TURN_FOR_TWISS, DEFAULT_STEPS_R_MATRIX


def get_R_matrix(
        line, particle_on_co,
        steps=None,
        start=None, end=None,
        num_turns=1,
        element_by_element=False,
        only_markers=False,
        symmetrize=True):
    import xpart

    if steps is None:
        steps = {}

    steps = _complete_steps_r_matrix_with_default(steps)

    if line.enable_time_dependent_vars:
        raise RuntimeError(
            'Time-dependent vars not supported in one-turn matrix computation')

    if isinstance(start, str):
        start = line._element_names_unique.index(start)

    if isinstance(end, str):
        end = line._element_names_unique.index(end)

    if start is not None and end is not None and start > end:
        raise ValueError('start > end')

    context = line._buffer.context

    particle_on_co = particle_on_co.copy(
                        _context=context)

    dx = steps["dx"]
    dpx = steps["dpx"]
    dy = steps["dy"]
    dpy = steps["dpy"]
    dzeta = steps["dzeta"]
    ddelta = steps["ddelta"]
    part_temp = xpart.build_particles(_context=context,
            particle_ref=particle_on_co, mode='shift',
            x  =    [0., dx,  0., 0.,  0.,    0.,     0., -dx,   0.,  0.,   0.,     0.,      0.],
            px =    [0., 0., dpx, 0.,  0.,    0.,     0.,  0., -dpx,  0.,   0.,     0.,      0.],
            y  =    [0., 0.,  0., dy,  0.,    0.,     0.,  0.,   0., -dy,   0.,     0.,      0.],
            py =    [0., 0.,  0., 0., dpy,    0.,     0.,  0.,   0.,  0., -dpy,     0.,      0.],
            zeta =  [0., 0.,  0., 0.,  0., dzeta,     0.,  0.,   0.,  0.,   0., -dzeta,      0.],
            delta = [0., 0.,  0., 0.,  0.,    0., ddelta,  0.,   0.,  0.,   0.,     0., -ddelta],
            )
    dpzeta = float(context.nparray_from_context_array(
        (part_temp.ptau[6] - part_temp.ptau[12])/2/part_temp.beta0[0]))
    if particle_on_co._xobject.at_element[0]>0:
        part_temp.s[:] = particle_on_co._xobject.s[0]
        part_temp.at_element[:] = particle_on_co._xobject.at_element[0]

    part_temp.at_turn = AT_TURN_FOR_TWISS

    if start is not None:
        assert element_by_element is False, 'Not yet implemented'
        assert num_turns == 1, 'Not yet implemented'
        assert num_turns == 1, 'Not yet implemented'
        assert end is not None
        line.track(part_temp, ele_start=start, ele_stop=end)
        if symmetrize:
            raise NotImplementedError
    elif particle_on_co._xobject.at_element[0]>0:
        assert element_by_element is False, 'Not yet implemented'
        assert num_turns == 1, 'Not yet implemented'
        assert symmetrize is False, 'Not yet implemented'
        i_start = particle_on_co._xobject.at_element[0]
        line.track(part_temp, ele_start=i_start)
        line.track(part_temp, num_elements=i_start)
    else:
        assert particle_on_co._xobject.at_element[0] == 0
        if element_by_element and num_turns != 1:
            raise NotImplementedError
        monitor_setting = 'ONE_TURN_EBE' if element_by_element else None
        line.track(part_temp, num_turns=num_turns,
                   turn_by_turn_monitor=monitor_setting)
        if symmetrize:
            raise NotImplementedError

    temp_mat = np.zeros(shape=(6, 13), dtype=np.float64)
    temp_mat[0, :] = context.nparray_from_context_array(part_temp.x)
    temp_mat[1, :] = context.nparray_from_context_array(part_temp.px)
    temp_mat[2, :] = context.nparray_from_context_array(part_temp.y)
    temp_mat[3, :] = context.nparray_from_context_array(part_temp.py)
    temp_mat[4, :] = context.nparray_from_context_array(part_temp.zeta)
    temp_mat[5, :] = context.nparray_from_context_array(
                                part_temp.ptau/part_temp.beta0) # pzeta

    RR = np.zeros(shape=(6, 6), dtype=np.float64)

    for jj, dd in enumerate([dx, dpx, dy, dpy, dzeta, dpzeta]):
        RR[:, jj] = (temp_mat[:, jj+1] - temp_mat[:, jj+1+6])/(2*dd)

    out = {'R_matrix': RR}
    out['steps_R_matrix'] = steps
    out['part_temp'] = part_temp

    if element_by_element:
        mon = line.record_last_track
        temp_mad_ebe = np.zeros(shape=(len(line._element_names_unique) + 1, 6, 13), dtype=np.float64)
        temp_mad_ebe[:, 0, :] = mon.x.T
        temp_mad_ebe[:, 1, :] = mon.px.T
        temp_mad_ebe[:, 2, :] = mon.y.T
        temp_mad_ebe[:, 3, :] = mon.py.T
        temp_mad_ebe[:, 4, :] = mon.zeta.T
        temp_mad_ebe[:, 5, :] = mon.ptau.T/mon.beta0.T

        RR_ebe = np.zeros(shape=(len(line._element_names_unique) + 1, 6, 6), dtype=np.float64)
        for jj, dd in enumerate([dx, dpx, dy, dpy, dzeta, dpzeta]):
            RR_ebe[:, :, jj] = (temp_mad_ebe[:, :, jj+1] - temp_mad_ebe[:, :, jj+1+6])/(2*dd)

        if only_markers:
            mask_twiss = line.tracker._get_twiss_mask_markers()
            mask_twiss[-1] = True # to include the "_end_point"

        out['R_matrix_ebe'] = RR_ebe
        out['mon_ebe'] = mon

    else:
        out['R_matrix_ebe'] = None

    return out


def compute_R_matrix(*args, **kwargs):
    warn(
        '`compute_R_matrix()` is deprecated and will be removed in future '
        'versions. Please use `get_R_matrix()` instead.'
        + DEPRECATION_INFO_PREP_1_0,
        FutureWarning,
    )
    return get_R_matrix(*args, **kwargs)


def _complete_steps_r_matrix_with_default(steps_R_matrix):
    if steps_R_matrix is not None:
        steps_in = steps_R_matrix.copy()
        for nn in steps_in.keys():
            assert nn in list(DEFAULT_STEPS_R_MATRIX.keys()) + ['adapted'], (
                '``steps_R_matrix`` can contain only ' +
                ' '.join(DEFAULT_STEPS_R_MATRIX.keys())
            )
        steps_R_matrix = DEFAULT_STEPS_R_MATRIX.copy()
        steps_R_matrix.update(steps_in)
    else:
        steps_R_matrix = DEFAULT_STEPS_R_MATRIX.copy()

    return steps_R_matrix

def get_T_matrix_line(line, start, end, particle_on_co=None,
                            steps=None):

    steps = _complete_steps_r_matrix_with_default(steps)

    if particle_on_co is None:
        tw = line.twiss(reverse=False)
        particle_on_co = tw.get_twiss_init(start).particle_on_co

    R_plus = {}
    R_minus = {}
    p_plus = {}
    p_minus = {}

    for kk in ['x', 'px', 'y', 'py', 'zeta', 'delta']:

        p_plus[kk] = particle_on_co.copy()
        setattr(p_plus[kk], kk, getattr(particle_on_co, kk) + steps['d' + kk])
        R_plus[kk] = line.get_R_matrix(
                            start=start, end=end,
                            particle_on_co=p_plus[kk])['R_matrix']

        p_minus[kk] = particle_on_co.copy()
        setattr(p_minus[kk], kk, getattr(particle_on_co, kk) - steps['d' + kk])
        R_minus[kk] = line.get_R_matrix(
                            start=start, end=end,
                            particle_on_co=p_minus[kk])['R_matrix']

    TT = np.zeros((6, 6, 6))
    TT[:, :, 0] = 0.5 * (R_plus['x'] - R_minus['x']) / (
        p_plus['x']._xobject.x[0] - p_minus['x']._xobject.x[0])
    TT[:, :, 1] = 0.5 * (R_plus['px'] - R_minus['px']) / (
        p_plus['px']._xobject.px[0] - p_minus['px']._xobject.px[0])
    TT[:, :, 2] = 0.5 * (R_plus['y'] - R_minus['y']) / (
        p_plus['y']._xobject.y[0] - p_minus['y']._xobject.y[0])
    TT[:, :, 3] = 0.5 * (R_plus['py'] - R_minus['py']) / (
        p_plus['py']._xobject.py[0] - p_minus['py']._xobject.py[0])
    TT[:, :, 4] = 0.5 * (R_plus['zeta'] - R_minus['zeta']) / (
        p_plus['zeta']._xobject.zeta[0] - p_minus['zeta']._xobject.zeta[0])
    TT[:, :, 5] = 0.5 * (R_plus['delta'] - R_minus['delta']) / (
        (p_plus['delta']._xobject.ptau[0] - p_minus['delta']._xobject.ptau[0])
        / p_plus['delta']._xobject.beta0[0])

    return TT


def compute_T_matrix_line(*args, **kwargs):
    warn(
        '`compute_T_matrix_line()` is deprecated and will be removed in future '
        'versions. Please use `get_T_matrix_line()` instead.'
        + DEPRECATION_INFO_PREP_1_0,
        FutureWarning,
    )
    return get_T_matrix_line(*args, **kwargs)
