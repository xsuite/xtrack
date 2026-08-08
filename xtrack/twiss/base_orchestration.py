# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from .base_computation import (
    _compute_base_twiss,
    _compute_base_twiss_after_explicit_init_completion,
)
from .base_init_acquisition import (
    _clear_twiss_init_inputs,
    _complete_init_for_base_twiss,
)
from .computation_plan import _plan_twiss_computation
from .constants import VARS_FOR_TWISS_INIT_GENERATION
from .element_indexing import _str_to_index
from .finalize import _finalize_twiss_result
from .twiss_table import TwissTable

import xtrack as xt  # To avoid circular imports


def _compute_twiss_with_prepared_line_context(data, input_kwargs):

    data = data.copy()
    data['zero_at_requested'] = data['zero_at']
    data['zero_at'] = None

    computation_plan = _plan_twiss_computation(data, data['init'])
    data['twiss_computation_plan'] = computation_plan
    route = computation_plan.route

    if route == 'periodic_one_turn_from_start':
        one_turn_kwargs = data.copy()
        one_turn_kwargs.pop('start')
        one_turn_plan = computation_plan.one_turn_propagation

        full_twiss = _compute_twiss_segment(one_turn_kwargs)
        first_part = full_twiss.rows[one_turn_plan.start:]
        second_part = full_twiss.rows[:one_turn_plan.start]
        twiss_res = TwissTable.concatenate([first_part, second_part])
        twiss_res.zero_at(twiss_res.name[0])
        twiss_res.name[-1] = '_end_point'
        twiss_res['periodic'] = True
        twiss_res['completed_init'] = full_twiss.completed_init

    elif route == 'open_one_turn_from_start':
        one_turn_kwargs = data.copy()
        one_turn_kwargs.pop('start')
        one_turn_kwargs.pop('end')
        one_turn_plan = computation_plan.one_turn_propagation

        first_part = _compute_twiss_segment_for_piece(
            kwargs=one_turn_kwargs,
            piece=one_turn_plan.first_piece,
            init=one_turn_kwargs['init'])
        second_part_init = first_part.get_twiss_init(
            one_turn_plan.transfer_init_at)
        second_part_init.element_name = (
            one_turn_plan.transfer_init_element_name)

        for field_name in VARS_FOR_TWISS_INIT_GENERATION:
            one_turn_kwargs.pop(field_name, None)
        one_turn_kwargs.pop('init')
        second_part = _compute_twiss_segment_for_piece(
            kwargs=one_turn_kwargs,
            piece=one_turn_plan.second_piece,
            init=second_part_init)
        second_part = second_part.rows[:-1]  # remove repeated element
        second_part.name[-1] = '_end_point'
        twiss_res = TwissTable.concatenate([first_part, second_part])
        twiss_res['completed_init'] = first_part.completed_init

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

        range_kwargs = data.copy()
        range_kwargs['init'] = full_periodic_init
        range_kwargs['init_at'] = None
        open_plan = computation_plan.open_propagation
        if open_plan.crosses_line_boundary:
            twiss_res = _handle_loop_around(
                range_kwargs, open_plan=open_plan)
        elif not open_plan.init_is_at_boundary:
            twiss_res = _handle_init_inside_range(
                range_kwargs, open_plan=open_plan)
        else:
            assert len(open_plan.pieces) == 1
            twiss_res = _compute_twiss_segment_for_piece(
                kwargs=range_kwargs,
                piece=open_plan.pieces[0],
                init=full_periodic_init)
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


def _compute_twiss_segment(kwargs, **overrides):

    segment_kwargs = kwargs.copy()
    segment_kwargs['disable_apertures'] = False
    segment_kwargs['freeze_longitudinal'] = False
    segment_kwargs['freeze_energy'] = False
    segment_kwargs['at_s'] = None
    segment_kwargs.update(overrides)

    return _compute_base_twiss(segment_kwargs)


def _compute_twiss_segment_for_piece(kwargs, piece, init):

    return _compute_twiss_segment(
        kwargs, start=piece.start, end=piece.end, init=init)


def _handle_loop_around(kwargs, open_plan):

    kwargs = kwargs.copy()
    init = kwargs.pop('init')
    start = kwargs.pop('start')
    end = kwargs.pop('end')
    line = kwargs['line']
    reverse = kwargs['reverse']

    if not reverse:
        assert _str_to_index(line, end) < _str_to_index(line, start), (
            'This function should not have been called')
    else:
        assert _str_to_index(line, end) > _str_to_index(line, start), (
            'This function should not have been called')
    assert open_plan.crosses_line_boundary
    assert len(open_plan.pieces) in (2, 3)

    if len(open_plan.pieces) == 3:
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
            twiss_tables = (first_side_table, third_table)
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
            twiss_tables = (first_table, second_side_table)
            completed_init = second_side_table.completed_init

        else:
            raise RuntimeError('Unexpected three-piece loop-around Twiss plan')

    else:
        first_piece, second_piece = open_plan.pieces
        init_index = _str_to_index(line, init.element_name)
        start_index = _str_to_index(line, start)
        end_index = _str_to_index(line, end)
        if not reverse and init_index >= start_index:
            init_is_in_first_piece = True
        elif not reverse and init_index <= end_index:
            init_is_in_first_piece = False
        elif reverse and init_index <= start_index:
            init_is_in_first_piece = True
        elif reverse and init_index >= end_index:
            init_is_in_first_piece = False
        else:
            raise RuntimeError(
                'Boundary conditions not at start or end of the specified range')

        if init_is_in_first_piece:
            first_table = _compute_twiss_segment_for_piece(
                kwargs=kwargs, piece=first_piece, init=init)
            boundary_init = first_table.get_twiss_init('_end_point')
            boundary_init.element_name = second_piece.start
            second_table = _compute_twiss_segment_for_piece(
                kwargs=kwargs, piece=second_piece, init=boundary_init)
            completed_init = first_table.completed_init
        else:
            second_table = _compute_twiss_segment_for_piece(
                kwargs=kwargs, piece=second_piece, init=init)
            boundary_init = second_table.get_twiss_init(second_piece.start)
            boundary_init.element_name = first_piece.end
            first_table = _compute_twiss_segment_for_piece(
                kwargs=kwargs, piece=first_piece, init=boundary_init)
            completed_init = second_table.completed_init

        twiss_tables = (first_table, second_table)

    return _combine_loop_around_twiss_tables(
        twiss_tables=twiss_tables,
        init=init,
        completed_init=completed_init)


def _handle_init_inside_range(kwargs, open_plan):

    kwargs = kwargs.copy()
    line = kwargs['line']
    start = kwargs.pop('start')
    end = kwargs.pop('end')
    init = kwargs.pop('init')
    reverse = kwargs.pop('reverse')

    init_element_name = init.element_name
    init_element = line.get(init_element_name)
    if isinstance(init_element, xt.Replica):
        init_element = init_element.resolve()
    if not isinstance(init_element, xt.Marker):
        raise ValueError(
            'The element at the initial position is not a Marker. '
            'This is not yet supported')

    if reverse:
        assert (_str_to_index(line, init_element_name)
                <= _str_to_index(line, start))
        assert (_str_to_index(line, init_element_name)
                >= _str_to_index(line, end))
    else:
        assert (_str_to_index(line, init_element_name)
                >= _str_to_index(line, start))
        assert (_str_to_index(line, init_element_name)
                <= _str_to_index(line, end))

    assert not open_plan.crosses_line_boundary
    assert not open_plan.init_is_at_boundary
    assert [piece.role for piece in open_plan.pieces] == [
        'before_init', 'after_init']

    first_table, second_table = tuple(
        _compute_twiss_segment(
            kwargs,
            start=piece.start,
            end=piece.end,
            init=init,
            reverse=reverse)
        for piece in open_plan.pieces)

    return _combine_init_inside_range_twiss_tables(
        first_table, second_table, init)


def _combine_loop_around_twiss_tables(
        twiss_tables, init, completed_init):

    init_element_name = init.element_name
    twiss_res = TwissTable.concatenate(twiss_tables)
    twiss_res.s -= twiss_res['s', init_element_name] - init.s
    twiss_res['completed_init'] = completed_init

    if 'mux' in twiss_res.keys():
        twiss_res.mux -= twiss_res['mux', init_element_name] - init.mux
        twiss_res.muy -= twiss_res['muy', init_element_name] - init.muy
        twiss_res.muzeta -= (
            twiss_res['muzeta', init_element_name] - init.muzeta)
    if 'dzeta' in twiss_res.keys():
        twiss_res.dzeta -= twiss_res['dzeta', init_element_name] - init.dzeta

    _remove_unsupported_phase_derivative_columns(twiss_res)
    twiss_res._data['loop_around'] = True
    _copy_common_metadata_from_tables(
        twiss_res=twiss_res, twiss_tables=twiss_tables)

    return twiss_res


def _combine_init_inside_range_twiss_tables(first_table, second_table, init):

    init_element_name = init.element_name
    twiss_res = TwissTable.concatenate([first_table, second_table])
    twiss_res['completed_init'] = first_table.completed_init

    twiss_res.s -= twiss_res['s', init_element_name] - init.s
    twiss_res.mux -= twiss_res['mux', init_element_name] - init.mux
    twiss_res.muy -= twiss_res['muy', init_element_name] - init.muy
    twiss_res.muzeta -= (
        twiss_res['muzeta', init_element_name] - init.muzeta)
    if 'dzeta' in twiss_res:
        twiss_res.dzeta -= twiss_res['dzeta', init_element_name] - init.dzeta

    _remove_unsupported_phase_derivative_columns(twiss_res)
    _copy_common_metadata_from_tables(
        twiss_res=twiss_res, twiss_tables=(first_table, second_table))

    return twiss_res


def _remove_unsupported_phase_derivative_columns(twiss_res):

    for column_name in ['dmux', 'dmuy']:
        if column_name in twiss_res.keys():
            twiss_res._data.pop(column_name)
            twiss_res._col_names.remove(column_name)


def _copy_common_metadata_from_tables(twiss_res, twiss_tables):

    for field_name in ['method', 'radiation_method', 'reference_frame']:
        values = [table[field_name] for table in twiss_tables]
        if all(value == values[0] for value in values[1:]):
            twiss_res._data[field_name] = values[0]
        else:
            twiss_res._data[field_name] = tuple(values)
