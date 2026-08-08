# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from .base_preparation import (
    _apply_base_twiss_reverse_range,
    _validate_base_twiss_boundary_init,
    _prepare_base_twiss_matrix_settings,
    _prepare_base_twiss_line_and_particle_ref,
    _validate_base_twiss_method,
    _validate_base_twiss_init_mode,
    _validate_base_twiss_open_momentum_offsets,
)
from .base_init_acquisition import (
    _acquire_base_twiss_init,
    _clear_twiss_init_inputs,
    _complete_init_for_base_twiss,
)
from .base_propagation import _propagate_twiss_from_init
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
from .computation_plan import _plan_twiss_computation
from .constants import VARS_FOR_TWISS_INIT_GENERATION
from .line_context import _set_twiss_periodic_mode
from .multiturn import _kwargs_for_multiturn_continuation


def _compute_base_twiss(data):
    """Run one normalized, non-composed Twiss computation."""

    data = data.copy()
    _set_missing_base_twiss_inputs(data)
    _set_twiss_periodic_mode(data)

    computation_plan = _plan_twiss_computation(data, data['init'])
    if computation_plan.route != 'base':
        raise RuntimeError(
            'A composed Twiss route reached the base segment engine: '
            f'{computation_plan.route}')

    data['twiss_computation_plan'] = computation_plan
    data['init'], data['completed_init'] = _complete_init_for_base_twiss(
        data=data)
    _clear_twiss_init_inputs(data)
    data['kwargs'] = data.copy()

    return _compute_base_twiss_after_explicit_init_completion(data)


def _compute_base_twiss_after_explicit_init_completion(data):
    """Acquire any periodic init, propagate, and finish one base result."""

    data = data.copy()

    data['start'], data['end'] = _apply_base_twiss_reverse_range(
        line=data['line'], start=data['start'], end=data['end'],
        reverse=data['reverse'])

    (data['matrix_responsiveness_tol'], data['matrix_stability_tol'],
        data['use_full_inverse']) = _prepare_base_twiss_matrix_settings(
            line=data['line'],
            radiation_method=data['radiation_method'],
            matrix_responsiveness_tol=data['matrix_responsiveness_tol'],
            matrix_stability_tol=data['matrix_stability_tol'],
            use_full_inverse=data['use_full_inverse'])

    data['line'], data['particle_ref'] = (
        _prepare_base_twiss_line_and_particle_ref(
            line=data['line'],
            particle_ref=data['particle_ref'],
            particle_on_co=data['particle_on_co'],
            co_guess=data['co_guess'],
            include_collective=data['include_collective']))
    data['method'] = _validate_base_twiss_method(data['method'])

    _validate_base_twiss_boundary_init(
        start=data['start'], init=data['init'])
    _validate_base_twiss_init_mode(init=data['init'])
    _validate_base_twiss_open_momentum_offsets(
        periodic=data['periodic'], delta0=data['delta0'], zeta0=data['zeta0'])

    data.update(_acquire_base_twiss_init(data))

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

    if not data['skip_global_quantities'] and not data['only_orbit']:
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


def _set_missing_base_twiss_inputs(data):

    fields_defaulting_to_none = (
        'start', 'end', 'init', 'init_at',
        *VARS_FOR_TWISS_INIT_GENERATION,
        'spin_x', 'spin_y', 'spin_z',
    )
    for field_name in fields_defaulting_to_none:
        data.setdefault(field_name, None)
