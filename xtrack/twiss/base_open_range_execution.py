# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from .element_indexing import _str_to_index
from .base_open_propagation import (
    _plan_init_inside_range_twiss_parts,
    _plan_loop_around_twiss_parts,
)
from .base_open_table_composition import (
    _combine_init_inside_range_twiss_tables,
    _combine_loop_around_twiss_tables,
)
from .segment_computation import (
    _compute_twiss_segment,
    _compute_twiss_segment_for_piece,
)

import xtrack as xt  # To avoid circular imports


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
