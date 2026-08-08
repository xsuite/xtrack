# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from .base_computation import (
    _clear_twiss_init_inputs,
    _complete_init_for_base_twiss,
    _compute_base_twiss,
    _compute_base_twiss_after_explicit_init_completion,
)
from .computation_plan import _plan_twiss_computation
from .constants import VARS_FOR_TWISS_INIT_GENERATION
from .element_indexing import _str_to_index
from .finalize import _finalize_twiss_result
from .open_propagation import (
    _plan_init_inside_range_twiss_parts,
    _plan_loop_around_twiss_parts,
)
from .open_table_composition import (
    _combine_init_inside_range_twiss_tables,
    _combine_loop_around_twiss_tables,
)

import xtrack as xt  # To avoid circular imports


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


def _propagate_full_periodic_init_over_range(kwargs, init, open_plan):

    range_kwargs = kwargs.copy()
    range_kwargs['init'] = init
    range_kwargs['init_at'] = None

    if open_plan.crosses_line_boundary:
        return _handle_loop_around(range_kwargs, open_plan=open_plan)

    if not open_plan.init_is_at_boundary:
        return _handle_init_inside_range(range_kwargs, open_plan=open_plan)

    assert len(open_plan.pieces) == 1
    return _compute_twiss_segment_for_piece(
        kwargs=range_kwargs, piece=open_plan.pieces[0], init=init)




def _compute_twiss_segment_for_piece(kwargs, piece, init):

    return _compute_twiss_segment(
        kwargs, start=piece.start, end=piece.end, init=init)


def _handle_loop_around(kwargs, open_plan=None):

    kwargs = kwargs.copy()

    init = kwargs.pop('init')
    start = kwargs.pop('start')
    end = kwargs.pop('end')

    line = kwargs['line']
    reverse = kwargs['reverse']

    if open_plan is not None and len(open_plan.pieces) == 3:
        twiss_tables, completed_init = _execute_three_piece_loop_around_plan(
            kwargs=kwargs, open_plan=open_plan, init=init)
        return _combine_loop_around_twiss_tables(
            twiss_tables, init, completed_init)

    plan = _plan_loop_around_twiss_parts(
        line=line, start=start, end=end, init=init, reverse=reverse,
        open_plan=open_plan)
    tw1, tw2, completed_init = _execute_loop_around_twiss_plan(
        kwargs=kwargs, plan=plan, init=init)

    return _combine_loop_around_twiss_tables(
        [tw1, tw2], init, completed_init)


def _execute_three_piece_loop_around_plan(kwargs, open_plan, init):

    first_piece, second_piece, third_piece = open_plan.pieces

    if first_piece.role == 'start_to_init':
        first_table = _compute_twiss_segment_for_piece(
            kwargs=kwargs, piece=first_piece, init=init)
        second_table = _compute_twiss_segment_for_piece(
            kwargs=kwargs, piece=second_piece, init=init)
        first_side_table = _combine_init_inside_range_twiss_tables(
            first_table, second_table, init)
        boundary_init = second_table.get_twiss_init('_end_point')
        boundary_init.element_name = third_piece.start
        third_table = _compute_twiss_segment_for_piece(
            kwargs=kwargs, piece=third_piece, init=boundary_init)
        loop_tables = (first_side_table, third_table)
        completed_init = first_side_table.completed_init

    elif first_piece.role == 'start_to_line_boundary':
        second_table = _compute_twiss_segment_for_piece(
            kwargs=kwargs, piece=second_piece, init=init)
        third_table = _compute_twiss_segment_for_piece(
            kwargs=kwargs, piece=third_piece, init=init)
        second_side_table = _combine_init_inside_range_twiss_tables(
            second_table, third_table, init)
        boundary_init = second_table.get_twiss_init(second_piece.start)
        boundary_init.element_name = first_piece.end
        first_table = _compute_twiss_segment_for_piece(
            kwargs=kwargs, piece=first_piece, init=boundary_init)
        loop_tables = (first_table, second_side_table)
        completed_init = second_side_table.completed_init

    else:
        raise RuntimeError('Unexpected three-piece loop-around Twiss plan')

    return loop_tables, completed_init


def _execute_loop_around_twiss_plan(kwargs, plan, init):

    if plan.init_piece_role == 'first_table_piece':
        tw1 = _compute_twiss_segment_for_piece(
            kwargs=kwargs, piece=plan.first_table_piece, init=init)
        twini_2 = tw1.get_twiss_init(at_element=plan.transfer_init_at)
        twini_2.element_name = plan.transfer_init_element_name
        tw2 = _compute_twiss_segment_for_piece(
            kwargs=kwargs, piece=plan.second_table_piece, init=twini_2)
        completed_init = tw1.completed_init
    elif plan.init_piece_role == 'second_table_piece':
        tw2 = _compute_twiss_segment_for_piece(
            kwargs=kwargs, piece=plan.second_table_piece, init=init)
        twini_1 = tw2.get_twiss_init(at_element=plan.transfer_init_at)
        twini_1.element_name = plan.transfer_init_element_name
        tw1 = _compute_twiss_segment_for_piece(
            kwargs=kwargs, piece=plan.first_table_piece, init=twini_1)
        completed_init = tw2.completed_init
    else:
        raise RuntimeError('Unexpected loop-around Twiss plan init piece')

    return tw1, tw2, completed_init


def _handle_init_inside_range(kwargs, open_plan=None):

    kwargs = kwargs.copy()
    line = kwargs['line']
    start = kwargs.pop('start')
    end = kwargs.pop('end')
    init = kwargs.pop('init')
    reverse = kwargs.pop('reverse')

    _assert_init_inside_range_is_supported(
        line=line, start=start, end=end, init=init, reverse=reverse)

    plan = _plan_init_inside_range_twiss_parts(
        line=line, start=start, end=end, init=init, reverse=reverse,
        open_plan=open_plan)
    tw1, tw2 = _execute_init_inside_range_twiss_plan(
        kwargs=kwargs, plan=plan, init=init, reverse=reverse)

    return _combine_init_inside_range_twiss_tables(tw1, tw2, init)


def _assert_init_inside_range_is_supported(line, start, end, init, reverse):

    ele_name_init = init.element_name
    ele_init = line.get(ele_name_init)
    if isinstance(ele_init, xt.Replica):
        ele_init = ele_init.resolve()
    if not isinstance(ele_init, xt.Marker):
        raise ValueError(
            'The element at the initial position is not a Marker. '
            'This is not yet supported')

    if reverse:
        assert _str_to_index(line, ele_name_init) <= _str_to_index(line, start)
        assert _str_to_index(line, ele_name_init) >= _str_to_index(line, end)
    else:
        assert _str_to_index(line, ele_name_init) >= _str_to_index(line, start)
        assert _str_to_index(line, ele_name_init) <= _str_to_index(line, end)


def _execute_init_inside_range_twiss_plan(kwargs, plan, init, reverse):

    return tuple(
        _compute_twiss_segment(
            kwargs,
            start=piece.start,
            end=piece.end,
            init=init,
            reverse=reverse)
        for piece in plan.pieces)


def _compute_twiss_segment(kwargs, **overrides):

    segment_kwargs = _kwargs_for_preflighted_twiss_segment(kwargs)
    segment_kwargs.update(overrides)

    return _compute_base_twiss(segment_kwargs)


def _kwargs_for_preflighted_twiss_segment(kwargs):

    segment_kwargs = kwargs.copy()
    segment_kwargs['disable_apertures'] = False
    segment_kwargs['freeze_longitudinal'] = False
    segment_kwargs['freeze_energy'] = False
    segment_kwargs['at_s'] = None

    return segment_kwargs




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


def _compute_one_turn_twiss_from_plan(kwargs, computation_plan):

    kwargs = kwargs.copy()
    kwargs.pop('start')
    route = computation_plan.route
    propagation_plan = computation_plan.one_turn_propagation

    if route == 'periodic_one_turn_from_start':
        return _compute_periodic_one_turn_twiss_from_start(
            kwargs=kwargs, plan=propagation_plan)

    if route == 'open_one_turn_from_start':
        return _compute_open_one_turn_twiss_from_start(
            kwargs=kwargs, plan=propagation_plan)

    raise RuntimeError(f'Unexpected one-turn Twiss route: {route}')


def _compute_periodic_one_turn_twiss_from_start(kwargs, plan):

    tw = _compute_twiss_segment(kwargs)
    t1 = tw.rows[plan.start:]
    t2 = tw.rows[:plan.start]
    out = xt.TwissTable.concatenate([t1, t2])
    out.zero_at(out.name[0])
    out.name[-1] = '_end_point'
    out['periodic'] = True
    out['completed_init'] = tw.completed_init
    return out


def _compute_open_one_turn_twiss_from_start(kwargs, plan):

    kwargs = kwargs.copy()
    kwargs.pop('end')

    t1o = _compute_twiss_segment_for_piece(
        kwargs=kwargs, piece=plan.first_piece, init=kwargs['init'])
    init_part2 = t1o.get_twiss_init(plan.transfer_init_at)
    init_part2.element_name = plan.transfer_init_element_name

    for kk in VARS_FOR_TWISS_INIT_GENERATION:
        kwargs.pop(kk, None)
    kwargs.pop('init')
    t2o = _compute_twiss_segment_for_piece(
        kwargs=kwargs, piece=plan.second_piece, init=init_part2)
    # remove repeated element
    t2o = t2o.rows[:-1]
    t2o.name[-1] = '_end_point'
    out = xt.TwissTable.concatenate([t1o, t2o])
    out['completed_init'] = t1o.completed_init
    return out

