# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from ..general import _print
from .element_indexing import _str_to_index


def _apply_base_twiss_reverse_range(line, start, end, reverse):

    if reverse:
        if start is not None and end is not None:
            assert (_str_to_index(line, start) >= _str_to_index(line, end)), (
                'start must be smaller than end in reverse mode')
        return end, start

    if start is not None and end is not None:
        assert _str_to_index(line, start) <= _str_to_index(line, end), (
            'start must be larger than end in forward mode')

    return start, end


def _validate_base_twiss_boundary_init(start, init):

    if start is not None and init is None:
        assert init is not None, (
            'init must be provided if start and end are used')


def _prepare_base_twiss_matrix_settings(
        line, radiation_method, matrix_responsiveness_tol,
        matrix_stability_tol, use_full_inverse):

    if matrix_responsiveness_tol is None:
        matrix_responsiveness_tol = line.matrix_responsiveness_tol
    if matrix_stability_tol is None:
        matrix_stability_tol = line.matrix_stability_tol

    if (line._radiation_model is not None
            and radiation_method != 'kick_as_co'):
        matrix_stability_tol = None
        if use_full_inverse is None:
            use_full_inverse = True

    return matrix_responsiveness_tol, matrix_stability_tol, use_full_inverse


def _prepare_base_twiss_line_and_particle_ref(
        line, particle_ref, particle_on_co, co_guess, include_collective):

    if particle_ref is None:
        if particle_on_co is not None:
            particle_ref = particle_on_co.copy()
        elif co_guess is None and hasattr(line, 'particle_ref'):
            particle_ref = line.particle_ref

    if line.iscollective and not include_collective:
        _print(
            'The line has collective elements.\n'
            'In the twiss computation collective elements are'
            ' replaced by drifts')
        line = line._get_non_collective_line()

    if particle_ref is None and co_guess is None:
        raise ValueError(
            "Either ``particle_ref`` or ``co_guess`` must be provided")

    return line, particle_ref


def _validate_base_twiss_method(method):

    if method is None:
        method = '6d'

    assert method in ['6d', '4d'], 'Method must be ``6d`` or ``4d``'

    return method


def _validate_base_twiss_init_mode(init):

    if isinstance(init, str):
        if init in ['preserve', 'preserve_start', 'preserve_end']:
            raise ValueError(f'init={init} not anymore supported')
        assert init == 'periodic' or 'full_periodic'


def _validate_base_twiss_open_momentum_offsets(periodic, delta0, zeta0):

    if not periodic:
        if delta0 is not None or zeta0 is not None:
            raise ValueError(
                'delta0 and zeta0 cannot be provided for open twiss')


def _periodic_solution_range_from_plan(acquisition_plan, start, end):

    assert acquisition_plan.computation_direction == 'forward'

    if acquisition_plan.scope == 'full_line':
        assert start is None and end is None
        return None, None

    if acquisition_plan.scope == 'requested_range':
        assert start is not None and end is not None
        return start, end

    raise RuntimeError(
        f'Unexpected periodic Twiss scope: {acquisition_plan.scope}')
