# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np
from scipy.constants import c as clight

from .chromatic_functions import _get_chromatic_functions
from .coupling_edw_teng import _get_coupling_elements_edwards_teng
from .multiturn import _extend_twiss_result_to_multiple_turns
from .periodic_solution import _find_periodic_solution
from .radiation import (
    _get_eneloss_and_damping_rates,
    _get_equilibrium_emittance_full,
    _get_equilibrium_emittance_kick_as_co,
)
from .ring_quantities import _add_ring_quantities
from .spin import _get_spin_polarization
from .strengths import _add_strengths_to_twiss_res

import xtrack as xt  # To avoid circular imports


def _add_periodic_solution_data_to_base_twiss(
        line, twiss_res, method, R_matrix, steps_R_matrix, RR_ebe,
        eigenvalues, Rot):

    twiss_res._data['R_matrix'] = R_matrix
    twiss_res._data['steps_R_matrix'] = steps_R_matrix
    twiss_res._data['steps_r_matrix'] = steps_R_matrix  # deprecated
    twiss_res._data['R_matrix_ebe'] = RR_ebe

    _add_ring_quantities(line=line, twiss_res=twiss_res, method=method)

    twiss_res._data['eigenvalues'] = eigenvalues.copy()
    twiss_res._data['rotation_matrix'] = Rot.copy()


def _add_chromatic_functions_to_twiss_result(
        line, twiss_res, init, chrom, periodic, only_orbit, delta_chrom,
        delta0, zeta0, steps_R_matrix, matrix_responsiveness_tol,
        matrix_stability_tol, symplectify, method, use_full_inverse,
        nemitt_x, nemitt_y, step_W_sigma, delta_disp, zeta_disp,
        start, end, num_turns, hide_thin_groups, only_markers,
        periodic_mode, include_collective):

    if only_orbit:
        return

    if not (chrom is True or (chrom is None and periodic)):
        return

    cols_chrom, scalars_chrom = _get_chromatic_functions(
        line=line,
        init=init,
        delta_chrom=delta_chrom,
        delta0=delta0,
        zeta0=zeta0,
        steps_R_matrix=steps_R_matrix,
        matrix_responsiveness_tol=matrix_responsiveness_tol,
        matrix_stability_tol=matrix_stability_tol,
        symplectify=symplectify,
        method=method,
        use_full_inverse=use_full_inverse,
        nemitt_x=nemitt_x,
        nemitt_y=nemitt_y,
        on_momentum_twiss_res=twiss_res,
        step_W_sigma=step_W_sigma,
        delta_disp=delta_disp,
        zeta_disp=zeta_disp,
        start=start,
        end=end,
        num_turns=num_turns,
        hide_thin_groups=hide_thin_groups,
        only_markers=only_markers,
        periodic=periodic,
        periodic_mode=periodic_mode,
        include_collective=include_collective,
    )
    twiss_res._data.update(cols_chrom)
    twiss_res._data.update(scalars_chrom)
    twiss_res._col_names += list(cols_chrom.keys())


def _add_radiation_analysis_to_twiss_result(
        line, twiss_res, radiation_analysis, only_orbit, method,
        steps_R_matrix, matrix_responsiveness_tol, start, end, nemitt_x,
        nemitt_y, step_W_sigma, zeta_shift, only_markers, radiation_method):

    if not radiation_analysis or only_orbit:
        return

    assert 'R_matrix' in twiss_res._data
    if method == '4d':
        raise ValueError('method="4d" not supported for radiation_analysis=True')

    with xt.line._preserve_config(line):
        with xt.line._preserve_track_flags(line):
            line.tracker.track_flags.XS_FLAG_SR_KICK_SAME_AS_FIRST = False
            line.config.XTRACK_SYNRAD_SCALE_SAME_AS_FIRST = False
            _, RR, _, _, _, RR_ebe = _find_periodic_solution(
                line=line,
                particle_ref=None,
                method='6d',
                particle_on_co=twiss_res.particle_on_co,
                co_search_settings=None,
                continue_on_closed_orbit_error=None,
                co_guess=None,
                steps_R_matrix=steps_R_matrix,
                symplectify=False,
                matrix_responsiveness_tol=matrix_responsiveness_tol,
                matrix_stability_tol=None,
                start=start, end=end,
                nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                step_W_sigma=step_W_sigma,
                delta0=None, zeta0=None, zeta_shift=zeta_shift,
                W_matrix=None, R_matrix=None,
                delta_disp=None,
                compute_R_element_by_element=True,
                only_markers=only_markers,
                factor_adapt_steps=0.03,
            )

    eneloss_damp_res = _get_eneloss_and_damping_rates(
        particle_on_co=twiss_res.particle_on_co, R_matrix=RR,
        W_matrix=twiss_res.W_matrix,
        px_co=twiss_res.px, py_co=twiss_res.py,
        ptau_co=twiss_res.ptau, t_rev0=twiss_res.t_rev0,
        line=line, radiation_method=radiation_method)
    twiss_res._data.update(eneloss_damp_res)

    for key in ['angle_rad', 'angle', 'rot_s_rad', 'length', 'radiation_flag']:
        if key not in twiss_res._data:
            values = line.attr[key]
            twiss_res[key] = np.concatenate([values, [values[0] * 0]])

    if radiation_method == 'kick_as_co':
        eq_emitts = _get_equilibrium_emittance_kick_as_co(
            twiss_res=twiss_res,
            damping_constants_turns=(
                eneloss_damp_res['damping_constants_turns']),
            radiation_method=radiation_method)
        twiss_res._data.update(eq_emitts)
    elif radiation_method == 'full':
        eq_emitts = _get_equilibrium_emittance_full(
            twiss_res=twiss_res,
            R_matrix_ebe=RR_ebe,
            radiation_method=radiation_method)
        twiss_res._data.update(eq_emitts)


def _apply_4d_longitudinal_result_convention(twiss_res, method):

    if method == '4d' and 'muzeta' in twiss_res._data:
        twiss_res.muzeta[:] = 0
        if 'qs' in twiss_res._data:
            twiss_res._data['qs'] = 0


def _set_twiss_result_values_at(twiss_res, values_at_element_exit):

    if values_at_element_exit:
        raise NotImplementedError
    twiss_res._data['values_at'] = 'entry'
    return twiss_res


def _add_strengths_and_radiation_integrals_to_twiss_result(
        line, twiss_res, strengths, radiation_integrals):

    if strengths or radiation_integrals:
        _add_strengths_to_twiss_res(twiss_res, line)
    if radiation_integrals:
        twiss_res._get_radiation_integrals(add_to_tw=True)


def _add_spin_polarization_to_twiss_result(
        line, twiss_res, method, polarization_analysis):

    if polarization_analysis:
        _get_spin_polarization(twiss_res, line, method)


def _add_edwards_teng_coupling_to_twiss_result(
        twiss_res, coupling_edw_teng, periodic, reverse):

    if not coupling_edw_teng:
        return
    if not periodic:
        raise ValueError(
            'Computing Edwards-Teng coupling elements is only supported for '
            'periodic lines.')
    if reverse:
        raise NotImplementedError(
            'Computing Edwards-Teng coupling elements in reverse mode is not '
            'yet implemented.')

    coupling_result = _get_coupling_elements_edwards_teng(
        W_matrix=twiss_res['W_matrix'],
        mux=twiss_res['mux'],
        muy=twiss_res['muy'],
        qx=twiss_res['qx'],
        qy=twiss_res['qy'])
    for key in coupling_result:
        twiss_res[key] = coupling_result[key]


def _add_base_twiss_metadata(line, twiss_res, method, radiation_method):

    twiss_res._data['method'] = method
    twiss_res._data['radiation_method'] = radiation_method
    twiss_res._data['reference_frame'] = 'proper'
    twiss_res._data['line_config'] = dict(line.config.copy())


def _reverse_twiss_result_if_needed(twiss_res, reverse):

    if reverse:
        return twiss_res.reverse()
    return twiss_res


def _add_measured_revolution_period_if_requested(twiss_res, search_for_t_rev):

    if not search_for_t_rev:
        return

    line_length = twiss_res.s[-1]
    beta0 = twiss_res.particle_on_co.beta0[0]
    t_rev_0 = line_length / clight / beta0
    twiss_res._data['t_rev'] = t_rev_0 - (
        twiss_res.zeta[-1] - twiss_res.zeta[0]) / (beta0 * clight)
    twiss_res._data['T_rev'] = twiss_res._data['t_rev']  # deprecated


def _extend_base_twiss_to_multiple_turns(twiss_res, num_turns, kwargs):

    return _extend_twiss_result_to_multiple_turns(
        twiss_res=twiss_res, num_turns=num_turns, kwargs=kwargs)


def _select_twiss_result_at_elements(twiss_res, at_elements):

    if at_elements is not None:
        return twiss_res.rows[at_elements]
    return twiss_res


def _add_periodicity_and_completed_init_to_twiss_result(
        twiss_res, periodic, completed_init):

    twiss_res['periodic'] = periodic
    twiss_res['completed_init'] = completed_init
    twiss_res._sort_col_names()


def _align_open_twiss_phases_with_init(twiss_res, init, reverse):

    if ((twiss_res._orientation == 'forward' and not reverse)
            or (twiss_res._orientation == 'backward' and reverse)):
        twiss_res.muzeta += init.muzeta - twiss_res.muzeta[0]
        if 'dzeta' in twiss_res._data:
            twiss_res.dzeta += init.dzeta - twiss_res.dzeta[0]
        if 'mux' in twiss_res._data:
            twiss_res.mux += init.mux - twiss_res.mux[0]
            twiss_res.muy += init.muy - twiss_res.muy[0]
    elif ((twiss_res._orientation == 'forward' and reverse)
            or (twiss_res._orientation == 'backward' and not reverse)):
        twiss_res.muzeta += init.muzeta - twiss_res.muzeta[-1]
        if 'dzeta' in twiss_res._data:
            twiss_res.dzeta += init.dzeta - twiss_res.dzeta[-1]
        if 'mux' in twiss_res._data:
            twiss_res.mux += init.mux - twiss_res.mux[-1]
            twiss_res.muy += init.muy - twiss_res.muy[-1]
