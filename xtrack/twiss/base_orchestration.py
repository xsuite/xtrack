# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from .base_computation import (
    _compute_base_twiss_after_explicit_init_completion,
)
from .base_init_acquisition import (
    _clear_twiss_init_inputs,
    _complete_init_for_base_twiss,
)
from .computation_plan import _plan_twiss_computation
from .finalize import _finalize_twiss_result
from .base_one_turn_execution import _compute_one_turn_twiss_from_plan
from .base_open_range_execution import (
    _handle_init_inside_range,
    _handle_loop_around,
    _propagate_full_periodic_init_over_range,
)
from .segment_computation import _compute_twiss_segment


def _compute_twiss_with_prepared_line_context(data, input_kwargs):

    data = data.copy()
    data['zero_at_requested'] = data['zero_at']
    data['zero_at'] = None

    computation_plan = _plan_twiss_computation(data, data['init'])
    data['twiss_computation_plan'] = computation_plan
    twiss_res = _compute_composed_twiss_before_init_completion(
        data=data, computation_plan=computation_plan)

    if twiss_res is None:
        data['init'], data['completed_init'] = _complete_init_for_base_twiss(
            data=data)
        _clear_twiss_init_inputs(data)

        twiss_res = _compute_composed_twiss_after_init_completion(
            data=data, computation_plan=computation_plan)

        if twiss_res is None:
            twiss_res = _compute_base_twiss_after_explicit_init_completion(
                data=data)

    return _finalize_twiss_result(
        twiss_res, input_kwargs, zero_at=data['zero_at_requested'])


def _compute_composed_twiss_before_init_completion(
        data, computation_plan):

    composed_twiss_res = None
    route = computation_plan.route

    if route in (
            'periodic_one_turn_from_start', 'open_one_turn_from_start'):
        call_kwargs = data.copy()
        composed_twiss_res = _compute_one_turn_twiss_from_plan(
            kwargs=call_kwargs,
            computation_plan=computation_plan,
        )

    elif route == 'full_periodic_range':
        call_kwargs = data.copy()
        periodic_kwargs = _prepare_kwargs_for_full_periodic_twiss(call_kwargs)
        full_periodic_init = _acquire_full_periodic_twiss_init(
            kwargs=periodic_kwargs,
            acquisition_plan=computation_plan.init_acquisition,
            start=data['start'],
        )
        composed_twiss_res = _propagate_full_periodic_init_over_range(
            kwargs=call_kwargs,
            init=full_periodic_init,
            open_plan=computation_plan.open_propagation,
        )
        if data['zero_at_requested'] is None:
            composed_twiss_res.zero_at(data['start'])

    elif route != 'base':
        raise RuntimeError(f'Unexpected Twiss computation route: {route}')

    return composed_twiss_res


def _compute_composed_twiss_after_init_completion(
        data, computation_plan):

    composed_twiss_res = None
    open_plan = _open_propagation_plan_after_init_completion(
        data=data, computation_plan=computation_plan)

    if open_plan is not None:
        call_kwargs = data.copy()

        if open_plan.crosses_line_boundary:
            composed_twiss_res = _handle_loop_around(
                call_kwargs, open_plan=open_plan)

        elif not open_plan.init_is_at_boundary:
            composed_twiss_res = _handle_init_inside_range(
                call_kwargs, open_plan=open_plan)

    return composed_twiss_res


def _open_propagation_plan_after_init_completion(data, computation_plan):

    init = data['init']
    open_plan = None

    if not data['periodic'] and not isinstance(init, str):
        open_plan = computation_plan.open_propagation

    return open_plan


def _prepare_kwargs_for_full_periodic_twiss(kwargs):

    kwargs = kwargs.copy()
    kwargs.pop('init')
    kwargs.pop('start')
    kwargs.pop('end')
    kwargs.pop('init_at')

    return kwargs


def _acquire_full_periodic_twiss_init(kwargs, acquisition_plan, start):
    """Compute the full periodic Twiss and extract the requested init."""

    assert acquisition_plan.source == 'full_periodic_solution'
    assert acquisition_plan.scope == 'full_line'
    assert acquisition_plan.computation_direction == 'forward'

    tw = _compute_twiss_segment(kwargs) # Periodic twiss of the full line

    return tw.get_twiss_init(acquisition_plan.init_at or start)
