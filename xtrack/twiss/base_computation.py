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


class _TwissBaseComputation:
    def __init__(self, data):
        self.__dict__.update(data)
        self.periodic_init_data = None

    def prepare_for_propagation_from_init(self):

        self._prepare_range_and_line()
        self._validate_base_request()
        acquisition_updates = _acquire_base_twiss_init(self.__dict__)
        self.__dict__.update(acquisition_updates)

    def _prepare_range_and_line(self):

        self.start, self.end = _apply_base_twiss_reverse_range(
            line=self.line, start=self.start, end=self.end,
            reverse=self.reverse)

        (self.matrix_responsiveness_tol, self.matrix_stability_tol,
            self.use_full_inverse) = _prepare_base_twiss_matrix_settings(
                line=self.line,
                radiation_method=self.radiation_method,
                matrix_responsiveness_tol=self.matrix_responsiveness_tol,
                matrix_stability_tol=self.matrix_stability_tol,
                use_full_inverse=self.use_full_inverse)

        self.line, self.particle_ref = _prepare_base_twiss_line_and_particle_ref(
            line=self.line,
            particle_ref=self.particle_ref,
            particle_on_co=self.particle_on_co,
            co_guess=self.co_guess,
            include_collective=self.include_collective)

        self.method = _validate_base_twiss_method(self.method)

    def _validate_base_request(self):

        _validate_base_twiss_boundary_init(start=self.start, init=self.init)
        _validate_base_twiss_init_mode(init=self.init)
        _validate_base_twiss_open_momentum_offsets(
            periodic=self.periodic, delta0=self.delta0, zeta0=self.zeta0)

    def init_for_only_twiss_init(self):

        if self.reverse:
            return self.init.reverse()
        return self.init

    def propagate_from_init(self):

        return _propagate_twiss_from_init(
            line=self.line,
            init=self.init,
            start=self.start,
            end=self.end,
            nemitt_x=self.nemitt_x,
            nemitt_y=self.nemitt_y,
            step_W_sigma=self.step_W_sigma,
            delta_disp=self.delta_disp,
            use_full_inverse=self.use_full_inverse,
            hide_thin_groups=self.hide_thin_groups,
            only_markers=self.only_markers,
            only_orbit=self.only_orbit,
            spin=self.spin,
            compute_lattice_functions=self.compute_lattice_functions,
            continue_if_lost=self._continue_if_lost,
            keep_tracking_data=self._keep_tracking_data,
            keep_initial_particles=self._keep_initial_particles,
            initial_particles=self._initial_particles,
            ebe_monitor=self._ebe_monitor)

    def finish_result(self, twiss_res):

        self.add_periodic_solution_data_to(twiss_res)
        self.add_chromatic_functions_to(twiss_res)
        self.add_radiation_analysis_to(twiss_res)
        _apply_4d_longitudinal_result_convention(
            twiss_res=twiss_res, method=self.method)
        twiss_res = self.set_values_at(twiss_res)
        self.add_strengths_and_radiation_integrals_to(twiss_res)
        self.add_spin_polarization_to(twiss_res)
        self.add_edwards_teng_coupling_to(twiss_res)
        self.add_metadata_to(twiss_res)
        twiss_res = self.reverse_result_if_needed(twiss_res)
        self.align_open_phases_with_init(twiss_res)
        self.add_measured_revolution_period_if_requested(twiss_res)
        twiss_res = self.extend_to_multiple_turns_if_needed(twiss_res)
        twiss_res = self.select_at_elements(twiss_res)
        self.add_periodicity_and_completed_init_to(twiss_res)

        return twiss_res

    def add_periodic_solution_data_to(self, twiss_res):

        if self.skip_global_quantities or self.only_orbit:
            return

        _add_periodic_solution_data_to_base_twiss(
            line=self.line,
            twiss_res=twiss_res,
            method=self.method,
            R_matrix=self.R_matrix,
            steps_R_matrix=self.steps_R_matrix,
            RR_ebe=self.RR_ebe,
            eigenvalues=self.eigenvalues,
            Rot=self.Rot)

    def add_chromatic_functions_to(self, twiss_res):

        _add_chromatic_functions_to_twiss_result(
            line=self.line,
            twiss_res=twiss_res,
            init=self.init,
            chrom=self.chrom,
            periodic=self.periodic,
            only_orbit=self.only_orbit,
            delta_chrom=self.delta_chrom,
            delta0=self.delta0,
            zeta0=self.zeta0,
            steps_R_matrix=self.steps_R_matrix,
            matrix_responsiveness_tol=self.matrix_responsiveness_tol,
            matrix_stability_tol=self.matrix_stability_tol,
            symplectify=self.symplectify,
            method=self.method,
            use_full_inverse=self.use_full_inverse,
            nemitt_x=self.nemitt_x,
            nemitt_y=self.nemitt_y,
            step_W_sigma=self.step_W_sigma,
            delta_disp=self.delta_disp,
            zeta_disp=self.zeta_disp,
            start=self.start,
            end=self.end,
            num_turns=self.num_turns,
            hide_thin_groups=self.hide_thin_groups,
            only_markers=self.only_markers,
            periodic_mode=self.periodic_mode,
            include_collective=self.include_collective)

    def add_radiation_analysis_to(self, twiss_res):

        _add_radiation_analysis_to_twiss_result(
            line=self.line,
            twiss_res=twiss_res,
            radiation_analysis=self.radiation_analysis,
            only_orbit=self.only_orbit,
            method=self.method,
            steps_R_matrix=self.steps_R_matrix,
            matrix_responsiveness_tol=self.matrix_responsiveness_tol,
            start=self.start,
            end=self.end,
            nemitt_x=self.nemitt_x,
            nemitt_y=self.nemitt_y,
            step_W_sigma=self.step_W_sigma,
            zeta_shift=self.zeta_shift,
            only_markers=self.only_markers,
            radiation_method=self.radiation_method)

    def set_values_at(self, twiss_res):

        return _set_twiss_result_values_at(
            twiss_res=twiss_res,
            values_at_element_exit=self.values_at_element_exit)

    def add_strengths_and_radiation_integrals_to(self, twiss_res):

        _add_strengths_and_radiation_integrals_to_twiss_result(
            line=self.line,
            twiss_res=twiss_res,
            strengths=self.strengths,
            radiation_integrals=self.radiation_integrals)

    def add_spin_polarization_to(self, twiss_res):

        _add_spin_polarization_to_twiss_result(
            line=self.line,
            twiss_res=twiss_res,
            method=self.method,
            polarization_analysis=self.polarization_analysis)

    def add_edwards_teng_coupling_to(self, twiss_res):

        _add_edwards_teng_coupling_to_twiss_result(
            twiss_res=twiss_res,
            coupling_edw_teng=self.coupling_edw_teng,
            periodic=self.periodic,
            reverse=self.reverse)

    def add_metadata_to(self, twiss_res):

        _add_base_twiss_metadata(
            line=self.line,
            twiss_res=twiss_res,
            method=self.method,
            radiation_method=self.radiation_method)

    def reverse_result_if_needed(self, twiss_res):

        return _reverse_twiss_result_if_needed(
            twiss_res=twiss_res, reverse=self.reverse)

    def align_open_phases_with_init(self, twiss_res):

        if not self.periodic and not self.only_orbit:
            _align_open_twiss_phases_with_init(
                twiss_res=twiss_res, init=self.init, reverse=self.reverse)

    def add_measured_revolution_period_if_requested(self, twiss_res):

        _add_measured_revolution_period_if_requested(
            twiss_res=twiss_res,
            search_for_t_rev=self.search_for_t_rev)

    def extend_to_multiple_turns_if_needed(self, twiss_res):

        if self.num_turns <= 1:
            return twiss_res

        kwargs = _kwargs_for_multiturn_continuation(
            self.kwargs, self.__dict__)
        return _extend_base_twiss_to_multiple_turns(
            twiss_res=twiss_res, num_turns=self.num_turns, kwargs=kwargs)

    def select_at_elements(self, twiss_res):

        return _select_twiss_result_at_elements(
            twiss_res=twiss_res, at_elements=self.at_elements)

    def add_periodicity_and_completed_init_to(self, twiss_res):

        _add_periodicity_and_completed_init_to_twiss_result(
            twiss_res=twiss_res,
            periodic=self.periodic,
            completed_init=self.completed_init)


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

    base_twiss = _TwissBaseComputation(data)
    base_twiss.prepare_for_propagation_from_init()

    if data['only_twiss_init']:
        assert data['periodic'], (
            '``only_twiss_init`` can only be used in periodic mode')
        return base_twiss.init_for_only_twiss_init()

    if data['only_markers'] and data['radiation_analysis']:
        raise NotImplementedError(
            '``only_markers`` not implemented for ``radiation_analysis``')

    twiss_res = base_twiss.propagate_from_init()
    return base_twiss.finish_result(twiss_res)


def _set_missing_base_twiss_inputs(data):

    fields_defaulting_to_none = (
        'start', 'end', 'init', 'init_at',
        *VARS_FOR_TWISS_INIT_GENERATION,
        'spin_x', 'spin_y', 'spin_z',
    )
    for field_name in fields_defaulting_to_none:
        data.setdefault(field_name, None)

