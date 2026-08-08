# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from .twiss_init import TwissInit
from .propagation import _propagate_twiss_from_init
from .base_result import (
    _add_periodic_solution_data_to_base_twiss,
    _add_chromatic_functions_to_twiss_result,
    _add_radiation_analysis_to_twiss_result,
    _apply_4d_longitudinal_result_convention,
    _set_twiss_result_values_at,
    _add_strengths_and_radiation_integrals_to_twiss_result,
    _add_spin_polarization_to_twiss_result,
    _add_edwards_teng_coupling_to_twiss_result,
    _add_base_twiss_metadata,
    _reverse_twiss_result_if_needed,
    _add_measured_revolution_period_if_requested,
    _extend_base_twiss_to_multiple_turns,
    _select_twiss_result_at_elements,
    _add_periodicity_and_completed_init_to_twiss_result,
    _align_open_twiss_phases_with_init,
)
from .element_indexing import _str_to_index
from .multiturn import _kwargs_for_multiturn_continuation


def _compute_base_twiss(data, **overrides):
    """Propagate from a concrete init and finish one non-composed result."""

    data = data.copy()
    # The public line context is already active. These options only control
    # context setup and must not be applied again for individual segments.
    data['disable_apertures'] = False
    data['freeze_longitudinal'] = False
    data['freeze_energy'] = False
    data['at_s'] = None
    data.update(overrides)

    if 'init' in overrides and 'completed_init' not in overrides:
        data['completed_init'] = data['init'].copy()
    assert isinstance(data['init'], TwissInit), (
        '_compute_base_twiss requires a concrete TwissInit')

    if data['reverse']:
        if data['start'] is not None and data['end'] is not None:
            assert (_str_to_index(data['line'], data['start'])
                    >= _str_to_index(data['line'], data['end'])), (
                'start must be smaller than end in reverse mode')
        data['start'], data['end'] = data['end'], data['start']
    elif data['start'] is not None and data['end'] is not None:
        assert (_str_to_index(data['line'], data['start'])
                <= _str_to_index(data['line'], data['end'])), (
            'start must be larger than end in forward mode')

    if data['only_twiss_init']:
        assert data['periodic'], (
            '``only_twiss_init`` can only be used in periodic mode')
        if data['reverse']:
            return data['init'].reverse()
        return data['init']

    if data['only_markers'] and data['radiation_analysis']:
        raise NotImplementedError(
            '``only_markers`` not implemented for ``radiation_analysis``')

    twiss_res = _propagate_twiss_from_init(
        line=data['line'],
        init=data['init'],
        start=data['start'],
        end=data['end'],
        nemitt_x=data['nemitt_x'],
        nemitt_y=data['nemitt_y'],
        step_W_sigma=data['step_W_sigma'],
        delta_disp=data['delta_disp'],
        use_full_inverse=data['use_full_inverse'],
        hide_thin_groups=data['hide_thin_groups'],
        only_markers=data['only_markers'],
        only_orbit=data['only_orbit'],
        spin=data['spin'],
        compute_lattice_functions=data['compute_lattice_functions'],
        continue_if_lost=data['_continue_if_lost'],
        keep_tracking_data=data['_keep_tracking_data'],
        keep_initial_particles=data['_keep_initial_particles'],
        initial_particles=data['_initial_particles'],
        ebe_monitor=data['_ebe_monitor'])

    if (data['periodic']
            and not data['skip_global_quantities']
            and not data['only_orbit']):
        _add_periodic_solution_data_to_base_twiss(
            line=data['line'],
            twiss_res=twiss_res,
            method=data['method'],
            R_matrix=data['R_matrix'],
            steps_R_matrix=data['steps_R_matrix'],
            RR_ebe=data['RR_ebe'],
            eigenvalues=data['eigenvalues'],
            Rot=data['Rot'])

    _add_chromatic_functions_to_twiss_result(
        line=data['line'],
        twiss_res=twiss_res,
        init=data['init'],
        chrom=data['chrom'],
        periodic=data['periodic'],
        only_orbit=data['only_orbit'],
        delta_chrom=data['delta_chrom'],
        delta0=data['delta0'],
        zeta0=data['zeta0'],
        steps_R_matrix=data['steps_R_matrix'],
        matrix_responsiveness_tol=data['matrix_responsiveness_tol'],
        matrix_stability_tol=data['matrix_stability_tol'],
        symplectify=data['symplectify'],
        method=data['method'],
        use_full_inverse=data['use_full_inverse'],
        nemitt_x=data['nemitt_x'],
        nemitt_y=data['nemitt_y'],
        step_W_sigma=data['step_W_sigma'],
        delta_disp=data['delta_disp'],
        zeta_disp=data['zeta_disp'],
        start=data['start'],
        end=data['end'],
        num_turns=data['num_turns'],
        hide_thin_groups=data['hide_thin_groups'],
        only_markers=data['only_markers'],
        periodic_mode=data['periodic_mode'],
        include_collective=data['include_collective'])

    _add_radiation_analysis_to_twiss_result(
        line=data['line'],
        twiss_res=twiss_res,
        radiation_analysis=data['radiation_analysis'],
        only_orbit=data['only_orbit'],
        method=data['method'],
        steps_R_matrix=data['steps_R_matrix'],
        matrix_responsiveness_tol=data['matrix_responsiveness_tol'],
        start=data['start'],
        end=data['end'],
        nemitt_x=data['nemitt_x'],
        nemitt_y=data['nemitt_y'],
        step_W_sigma=data['step_W_sigma'],
        zeta_shift=data['zeta_shift'],
        only_markers=data['only_markers'],
        radiation_method=data['radiation_method'])

    _apply_4d_longitudinal_result_convention(
        twiss_res=twiss_res, method=data['method'])
    twiss_res = _set_twiss_result_values_at(
        twiss_res=twiss_res,
        values_at_element_exit=data['values_at_element_exit'])

    _add_strengths_and_radiation_integrals_to_twiss_result(
        line=data['line'],
        twiss_res=twiss_res,
        strengths=data['strengths'],
        radiation_integrals=data['radiation_integrals'])
    _add_spin_polarization_to_twiss_result(
        line=data['line'],
        twiss_res=twiss_res,
        method=data['method'],
        polarization_analysis=data['polarization_analysis'])
    _add_edwards_teng_coupling_to_twiss_result(
        twiss_res=twiss_res,
        coupling_edw_teng=data['coupling_edw_teng'],
        periodic=data['periodic'],
        reverse=data['reverse'])
    _add_base_twiss_metadata(
        line=data['line'],
        twiss_res=twiss_res,
        method=data['method'],
        radiation_method=data['radiation_method'])

    twiss_res = _reverse_twiss_result_if_needed(
        twiss_res=twiss_res, reverse=data['reverse'])
    if not data['periodic'] and not data['only_orbit']:
        _align_open_twiss_phases_with_init(
            twiss_res=twiss_res, init=data['init'], reverse=data['reverse'])
    _add_measured_revolution_period_if_requested(
        twiss_res=twiss_res,
        search_for_t_rev=data['search_for_t_rev'])

    if data['num_turns'] > 1:
        multiturn_kwargs = _kwargs_for_multiturn_continuation(
            data['kwargs'], data)
        twiss_res = _extend_base_twiss_to_multiple_turns(
            twiss_res=twiss_res,
            num_turns=data['num_turns'],
            kwargs=multiturn_kwargs)

    twiss_res = _select_twiss_result_at_elements(
        twiss_res=twiss_res, at_elements=data['at_elements'])
    _add_periodicity_and_completed_init_to_twiss_result(
        twiss_res=twiss_res,
        periodic=data['periodic'],
        completed_init=data['completed_init'])

    return twiss_res
