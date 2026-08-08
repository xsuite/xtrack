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
    route = computation_plan.route

    if route in (
            'periodic_one_turn_from_start', 'open_one_turn_from_start'):
        twiss_res = _compute_one_turn_twiss_from_plan(
            kwargs=data.copy(),
            computation_plan=computation_plan,
        )

    elif route == 'full_periodic_range':
        acquisition_plan = computation_plan.init_acquisition
        assert acquisition_plan.source == 'full_periodic_solution'
        assert acquisition_plan.scope == 'full_line'
        assert acquisition_plan.computation_direction == 'forward'

        periodic_kwargs = data.copy()
        periodic_kwargs.pop('init')
        periodic_kwargs.pop('start')
        periodic_kwargs.pop('end')
        periodic_kwargs.pop('init_at')
        full_periodic_twiss = _compute_twiss_segment(periodic_kwargs)
        full_periodic_init = full_periodic_twiss.get_twiss_init(
            acquisition_plan.init_at or data['start'])

        twiss_res = _propagate_full_periodic_init_over_range(
            kwargs=data.copy(),
            init=full_periodic_init,
            open_plan=computation_plan.open_propagation,
        )
        if data['zero_at_requested'] is None:
            twiss_res.zero_at(data['start'])

    elif route == 'base':
        data['init'], data['completed_init'] = _complete_init_for_base_twiss(
            data=data)
        _clear_twiss_init_inputs(data)

        open_plan = None
        if not data['periodic'] and not isinstance(data['init'], str):
            open_plan = computation_plan.open_propagation

        if open_plan is not None and open_plan.crosses_line_boundary:
            twiss_res = _handle_loop_around(
                data.copy(), open_plan=open_plan)
        elif open_plan is not None and not open_plan.init_is_at_boundary:
            twiss_res = _handle_init_inside_range(
                data.copy(), open_plan=open_plan)
        else:
            twiss_res = _compute_base_twiss_after_explicit_init_completion(
                data=data)

    else:
        raise RuntimeError(f'Unexpected Twiss computation route: {route}')

    return _finalize_twiss_result(
        twiss_res, input_kwargs, zero_at=data['zero_at_requested'])
