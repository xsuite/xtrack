# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np
from scipy.constants import c as clight

from .chromatic_functions import _get_chromatic_functions
from .coupling_edw_teng import _get_coupling_elements_edwards_teng
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


def _add_periodic_solution_data_to_base_twiss(twiss_config, twiss_res):

    twiss_res._data['R_matrix'] = twiss_config['R_matrix']
    twiss_res._data['steps_R_matrix'] = twiss_config['steps_R_matrix']
    twiss_res._data['steps_r_matrix'] = twiss_config['steps_R_matrix']  # deprecated
    twiss_res._data['R_matrix_ebe'] = twiss_config['RR_ebe']

    _add_ring_quantities(
        line=twiss_config['line'], twiss_res=twiss_res, method=twiss_config['method'])

    twiss_res._data['eigenvalues'] = twiss_config['eigenvalues'].copy()
    twiss_res._data['rotation_matrix'] = twiss_config['Rot'].copy()


def _add_chromatic_functions_to_twiss_result(twiss_config, twiss_res):

    if twiss_config['only_orbit']:
        return

    if not (twiss_config['chrom'] is True
            or (twiss_config['chrom'] is None and twiss_config['periodic'])):
        return

    cols_chrom, scalars_chrom = _get_chromatic_functions(
        line=twiss_config['line'],
        init=twiss_config['init'],
        delta_chrom=twiss_config['delta_chrom'],
        delta0=twiss_config['delta0'],
        zeta0=twiss_config['zeta0'],
        steps_R_matrix=twiss_config['steps_R_matrix'],
        matrix_responsiveness_tol=twiss_config['matrix_responsiveness_tol'],
        matrix_stability_tol=twiss_config['matrix_stability_tol'],
        symplectify=twiss_config['symplectify'],
        method=twiss_config['method'],
        use_full_inverse=twiss_config['use_full_inverse'],
        nemitt_x=twiss_config['nemitt_x'],
        nemitt_y=twiss_config['nemitt_y'],
        on_momentum_twiss_res=twiss_res,
        step_W_sigma=twiss_config['step_W_sigma'],
        delta_disp=twiss_config['delta_disp'],
        zeta_disp=twiss_config['zeta_disp'],
        start=twiss_config['start'],
        end=twiss_config['end'],
        num_turns=twiss_config['num_turns'],
        hide_thin_groups=twiss_config['hide_thin_groups'],
        only_markers=twiss_config['only_markers'],
        periodic=twiss_config['periodic'],
        periodic_mode=twiss_config['periodic_mode'],
        include_collective=twiss_config['include_collective'],
    )
    twiss_res._data.update(cols_chrom)
    twiss_res._data.update(scalars_chrom)
    twiss_res._col_names += list(cols_chrom.keys())


def _add_radiation_analysis_to_twiss_result(twiss_config, twiss_res):

    if not twiss_config['radiation_analysis'] or twiss_config['only_orbit']:
        return

    assert 'R_matrix' in twiss_res._data
    if twiss_config['method'] == '4d':
        raise ValueError('method="4d" not supported for radiation_analysis=True')

    line = twiss_config['line']
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
                steps_R_matrix=twiss_config['steps_R_matrix'],
                symplectify=False,
                matrix_responsiveness_tol=twiss_config['matrix_responsiveness_tol'],
                matrix_stability_tol=None,
                start=twiss_config['start'], end=twiss_config['end'],
                nemitt_x=twiss_config['nemitt_x'], nemitt_y=twiss_config['nemitt_y'],
                step_W_sigma=twiss_config['step_W_sigma'],
                delta0=None, zeta0=None, zeta_shift=twiss_config['zeta_shift'],
                W_matrix=None, R_matrix=None,
                delta_disp=None,
                compute_R_element_by_element=True,
                only_markers=twiss_config['only_markers'],
                factor_adapt_steps=0.03,
            )

    eneloss_damp_res = _get_eneloss_and_damping_rates(
        particle_on_co=twiss_res.particle_on_co, R_matrix=RR,
        W_matrix=twiss_res.W_matrix,
        px_co=twiss_res.px, py_co=twiss_res.py,
        ptau_co=twiss_res.ptau, t_rev0=twiss_res.t_rev0,
        line=line, radiation_method=twiss_config['radiation_method'])
    twiss_res._data.update(eneloss_damp_res)

    for key in ['angle_rad', 'angle', 'rot_s_rad', 'length', 'radiation_flag']:
        if key not in twiss_res._data:
            values = line.attr[key]
            twiss_res[key] = np.concatenate([values, [values[0] * 0]])

    if twiss_config['radiation_method'] == 'kick_as_co':
        eq_emitts = _get_equilibrium_emittance_kick_as_co(
            twiss_res=twiss_res,
            damping_constants_turns=(
                eneloss_damp_res['damping_constants_turns']),
            radiation_method=twiss_config['radiation_method'])
        twiss_res._data.update(eq_emitts)
    elif twiss_config['radiation_method'] == 'full':
        eq_emitts = _get_equilibrium_emittance_full(
            twiss_res=twiss_res,
            R_matrix_ebe=RR_ebe,
            radiation_method=twiss_config['radiation_method'])
        twiss_res._data.update(eq_emitts)


def _apply_4d_longitudinal_result_convention(twiss_config, twiss_res):

    if twiss_config['method'] == '4d' and 'muzeta' in twiss_res._data:
        twiss_res.muzeta[:] = 0
        if 'qs' in twiss_res._data:
            twiss_res._data['qs'] = 0


def _set_twiss_result_values_at(twiss_config, twiss_res):

    if twiss_config['values_at_element_exit']:
        raise NotImplementedError
    twiss_res._data['values_at'] = 'entry'
    return twiss_res


def _add_strengths_and_radiation_integrals_to_twiss_result(twiss_config, twiss_res):

    if twiss_config['strengths'] or twiss_config['radiation_integrals']:
        _add_strengths_to_twiss_res(twiss_res, twiss_config['line'])
    if twiss_config['radiation_integrals']:
        twiss_res._get_radiation_integrals(add_to_tw=True)


def _add_spin_polarization_to_twiss_result(twiss_config, twiss_res):

    if twiss_config['polarization_analysis']:
        _get_spin_polarization(twiss_res, twiss_config['line'], twiss_config['method'])


def _add_edwards_teng_coupling_to_twiss_result(twiss_config, twiss_res):

    if not twiss_config['coupling_edw_teng']:
        return
    if not twiss_config['periodic']:
        raise ValueError(
            'Computing Edwards-Teng coupling elements is only supported for '
            'periodic lines.')
    if twiss_config['reverse']:
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


def _add_base_twiss_metadata(twiss_config, twiss_res):

    twiss_res._data['method'] = twiss_config['method']
    twiss_res._data['radiation_method'] = twiss_config['radiation_method']
    twiss_res._data['reference_frame'] = 'proper'
    twiss_res._data['line_config'] = dict(twiss_config['line'].config.copy())


def _reverse_twiss_result_if_needed(twiss_config, twiss_res):

    if twiss_config['reverse']:
        return twiss_res.reverse()
    return twiss_res


def _add_measured_revolution_period_if_requested(twiss_config, twiss_res):

    if not twiss_config['search_for_t_rev']:
        return

    line_length = twiss_res.s[-1]
    beta0 = twiss_res.particle_on_co.beta0[0]
    t_rev_0 = line_length / clight / beta0
    twiss_res._data['t_rev'] = t_rev_0 - (
        twiss_res.zeta[-1] - twiss_res.zeta[0]) / (beta0 * clight)
    twiss_res._data['T_rev'] = twiss_res._data['t_rev']  # deprecated


def _align_open_twiss_phases_with_init(twiss_config, twiss_res):

    init = twiss_config['init']
    reverse = twiss_config['reverse']
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
