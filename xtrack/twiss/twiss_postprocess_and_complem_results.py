# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np
import xobjects as xo
from scipy.constants import c as clight

from .chromatic_functions import _get_chromatic_functions, trapz
from .coupling_edw_teng import _get_coupling_elements_edwards_teng
from .periodic_solution import _find_periodic_solution
from .radiation import (
    _get_eneloss_and_damping_rates,
    _get_equilibrium_emittance_full,
    _get_equilibrium_emittance_kick_as_co,
)
from .spin import _add_spin_polarization
from .strengths import _add_strengths_to_twiss_res

import xtrack as xt  # To avoid circular imports


def _add_periodic_solution_data_to_twiss_result(twiss_config, twiss_res):

    twiss_res._data['R_matrix'] = twiss_config['R_matrix']
    twiss_res._data['steps_R_matrix'] = twiss_config['steps_R_matrix']
    twiss_res._data['steps_r_matrix'] = twiss_config['steps_R_matrix']  # deprecated
    twiss_res._data['R_matrix_ebe'] = twiss_config['RR_ebe']

    line = twiss_config['line']
    method = twiss_config['method']
    s_vect = twiss_res['s']
    line_length = line.tracker._tracker_data_base.line_length
    part_on_co = twiss_res['particle_on_co']
    W_matrix = twiss_res['W_matrix']

    beta0 = part_on_co._xobject.beta0[0]
    gamma0 = part_on_co._xobject.gamma0[0]
    t_rev0 = line_length / clight / beta0
    bets0 = W_matrix[0, 4, 4]**2 + W_matrix[0, 4, 5]**2

    # Compute slip factor.
    if method == '6d':
        RR = twiss_res['R_matrix']
        dz_test = 1e-3  # All linear, so the value does not matter.
        xx = np.linalg.solve(
            RR - np.eye(6), np.array([0, 0, 0, 0, dz_test, 0]))
        delta_test = xx[5]
    elif method == '4d':
        RR = twiss_res['R_matrix'].copy()
        solve_mat = RR - np.eye(6)
        solve_mat[4, :] = np.array([0, 0, 0, 0, 1, 0])  # dummy
        solve_mat[5, :] = np.array([0, 0, 0, 0, 0, 1])  # delta
        delta_test = 1e-3  # All linear, so the value does not matter.
        xx = np.linalg.solve(
            solve_mat, np.array([0, 0, 0, 0, 0, delta_test]))
        # Measure slippage on the original matrix.
        xx_out = twiss_res['R_matrix'] @ xx
        dz_test = xx_out[4] - xx[4]

    slip_factor_dzeta_ddelta = dz_test / delta_test

    if line_length > 0:
        slip_factor = -slip_factor_dzeta_ddelta / line_length
        momentum_compaction_factor = slip_factor + 1 / gamma0**2
    else:
        slip_factor = np.nan
        momentum_compaction_factor = np.nan

    if slip_factor_dzeta_ddelta > 0:  # below transition
        bets0 = -bets0

    twiss_res._data.update({
        'bets0': bets0,
        'line_length': line_length,
        'circumference': line_length,  # deprecated
        'T_rev0': t_rev0,  # deprecated
        't_rev0': t_rev0,
        'particle_on_co': part_on_co.copy(_context=xo.context_default),
        'gamma0': gamma0,
        'beta0': beta0,
        'p0c': part_on_co._xobject.p0c[0],
        'slip_factor': slip_factor,
        'momentum_compaction_factor': momentum_compaction_factor,
        'slip_factor_dz_ddelta': slip_factor_dzeta_ddelta,  # deprecated
        'slip_factor_dzeta_ddelta': slip_factor_dzeta_ddelta,
    })

    if hasattr(part_on_co, '_fsolve_info'):
        twiss_res.particle_on_co._fsolve_info = part_on_co._fsolve_info
    else:
        twiss_res.particle_on_co._fsolve_info = None

    if 'mux' in twiss_res._data:  # Lattice functions are available.
        mux = twiss_res['mux']
        muy = twiss_res['muy']

        # Coupling
        # from Y. Luo et al., "Possible phase loop for the global betatron decoupling",
        # C-A/AP/#174, https://www.agsrhichome.bnl.gov//AP/ap_notes/ap_note_174.pdf
        w11 = W_matrix[:, 0, 0]
        w13 = W_matrix[:, 0, 2]
        w14 = W_matrix[:, 0, 3]
        w31 = W_matrix[:, 2, 0]
        w32 = W_matrix[:, 2, 1]
        w33 = W_matrix[:, 2, 2]

        c_r1 = np.sqrt(w31**2 + w32**2) / w11
        c_r2 = np.sqrt(w13**2 + w14**2) / w33
        c_phi1 = np.arctan2(w32, w31)
        c_phi2 = np.arctan2(w14, w13)

        # Coupling (https://arxiv.org/pdf/2005.02753.pdf)
        # R. Jones, Measuring Tune, Chromaticity and Coupling,
        # Proceedings of the 2018 CERN Accelerator School.
        cmin_arr = (2 * np.sqrt(c_r1 * c_r2)
                    * np.abs(np.mod(mux[-1], 1) - np.mod(muy[-1], 1))
                    / (1 + c_r1 * c_r2))
        if line_length > 0:
            c_minus = trapz(cmin_arr, s_vect) / line_length
        else:
            c_minus = np.mean(cmin_arr)

        c_minus_cplx = c_minus * np.exp(1j * c_phi1)
        c_minus_re = np.real(c_minus_cplx)
        c_minus_im = np.imag(c_minus_cplx)
        c_minus_local = cmin_arr * np.exp(1j * c_phi1)

        qs = np.abs(twiss_res['muzeta'][-1])

        # Scalars
        twiss_res._data.update({
            'qx': mux[-1], 'qy': muy[-1], 'qs': qs,
            'c_minus': c_minus,
            'c_minus_re_0': c_minus_re[0], 'c_minus_im_0': c_minus_im[0],
            'c_minus_local': c_minus_local,
        })

        # Coupling columns
        twiss_res['c_minus_re'] = c_minus_re
        twiss_res['c_minus_im'] = c_minus_im
        twiss_res['c_r1'] = c_r1
        twiss_res['c_r2'] = c_r2
        twiss_res['c_phi1'] = c_phi1
        twiss_res['c_phi2'] = c_phi2

    twiss_res._data['eigenvalues'] = twiss_config['eigenvalues'].copy()
    twiss_res._data['rotation_matrix'] = twiss_config['Rot'].copy()


def _add_chromatic_functions_to_twiss_result(twiss_config, twiss_res):

    if twiss_config['only_orbit']:
        return

    if not (twiss_config['chrom'] is True
            or (twiss_config['chrom'] is None and twiss_config['periodic'])):
        return

    cols_chrom, scalars_chrom = _get_chromatic_functions(
        twiss_config, on_momentum_twiss_res=twiss_res)
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
        _add_spin_polarization(
            twiss_res, twiss_config['line'], twiss_config['method'])


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
