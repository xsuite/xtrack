# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from .element_indexing import _str_to_index
from .twiss_table import TwissTable

import xtrack as xt  # To avoid circular imports


def _handle_init_inside_range_and_line_wrap(
        kwargs, crosses_line_boundary, one_turn_from_start=False, *,
        compute_base_twiss):

    if not crosses_line_boundary:
        kwargs = kwargs.copy()
        line = kwargs['line']
        start = kwargs.pop('start')
        end = kwargs.pop('end')
        init = kwargs.pop('init')
        reverse = kwargs['reverse']

        # Bidirectional propagation from an interior init is supported at
        # markers.
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

        # Propagate both sides from the same init, then restore one continuous
        # table.
        first_table, second_table = tuple(
            _compute_twiss_range_piece(
                kwargs,
                start=piece_start,
                end=piece_end,
                init=init,
                compute_base_twiss=compute_base_twiss)
            for piece_start, piece_end in (
                (start, init_element_name),
                (init_element_name, end),
            ))

        return _combine_init_inside_range_twiss_tables(
            first_table, second_table, init)

    kwargs = kwargs.copy()
    init = kwargs.pop('init')
    start = kwargs.pop('start')
    end = kwargs.pop('end')
    line = kwargs['line']
    reverse = kwargs['reverse']

    # Confirm that the requested traversal crosses the physical line boundary.
    if one_turn_from_start:
        assert start == end
    elif not reverse:
        assert _str_to_index(line, end) < _str_to_index(line, start), (
            'This function should not have been called')
    else:
        assert _str_to_index(line, end) > _str_to_index(line, start), (
            'This function should not have been called')

    if reverse:
        line_boundary_end = line._element_names_unique[0]
        line_boundary_start = line._element_names_unique[-1]
    else:
        line_boundary_end = line._element_names_unique[-1]
        line_boundary_start = line._element_names_unique[0]

    if one_turn_from_start:
        # A start without an end requests one complete turn. Propagate across
        # the physical line boundary, then replace the repeated start row with
        # the conventional final _end_point row.
        first_table = _compute_twiss_range_piece(
            kwargs, start=start, end=line_boundary_end, init=init,
            compute_base_twiss=compute_base_twiss)
        boundary_init = first_table.get_twiss_init('_end_point')
        boundary_init.element_name = line_boundary_start
        second_table = _compute_twiss_range_piece(
            kwargs,
            start=line_boundary_start,
            end=start,
            init=boundary_init,
            compute_base_twiss=compute_base_twiss)
        second_table = second_table.rows[:-1]
        second_table.name[-1] = '_end_point'
        twiss_res = TwissTable.concatenate([first_table, second_table])
        twiss_res['completed_init'] = first_table.completed_init
        return twiss_res

    init_element_name = init.element_name
    init_index = _str_to_index(line, init_element_name)
    start_index = _str_to_index(line, start)
    end_index = _str_to_index(line, end)

    if init_element_name not in (start, end):
        # Build the side containing the init in both directions, then transfer
        # its boundary conditions across the physical end of the line.
        init_is_after_start = (
            (not reverse and init_index >= start_index)
            or (reverse and init_index <= start_index))

        if init_is_after_start:
            first_table = _compute_twiss_range_piece(
                kwargs, start=start, end=init_element_name, init=init,
                compute_base_twiss=compute_base_twiss)
            second_table = _compute_twiss_range_piece(
                kwargs, start=init_element_name, end=line_boundary_end,
                init=init,
                compute_base_twiss=compute_base_twiss)
            first_side_table = _combine_init_inside_range_twiss_tables(
                first_table, second_table, init)
            boundary_init = second_table.get_twiss_init('_end_point')
            boundary_init.element_name = line_boundary_start
            third_table = _compute_twiss_range_piece(
                kwargs, start=line_boundary_start, end=end,
                init=boundary_init,
                compute_base_twiss=compute_base_twiss)
            twiss_tables = (first_side_table, third_table)
            completed_init = first_side_table.completed_init

        else:
            second_table = _compute_twiss_range_piece(
                kwargs, start=line_boundary_start, end=init_element_name,
                init=init,
                compute_base_twiss=compute_base_twiss)
            third_table = _compute_twiss_range_piece(
                kwargs, start=init_element_name, end=end, init=init,
                compute_base_twiss=compute_base_twiss)
            second_side_table = _combine_init_inside_range_twiss_tables(
                second_table, third_table, init)
            boundary_init = second_table.get_twiss_init(line_boundary_start)
            boundary_init.element_name = line_boundary_end
            first_table = _compute_twiss_range_piece(
                kwargs, start=start, end=line_boundary_end,
                init=boundary_init,
                compute_base_twiss=compute_base_twiss)
            twiss_tables = (first_table, second_side_table)
            completed_init = second_side_table.completed_init

    else:
        # With a boundary init, propagate its side first and transfer across.
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
            first_table = _compute_twiss_range_piece(
                kwargs, start=start, end=line_boundary_end, init=init,
                compute_base_twiss=compute_base_twiss)
            boundary_init = first_table.get_twiss_init('_end_point')
            boundary_init.element_name = line_boundary_start
            second_table = _compute_twiss_range_piece(
                kwargs, start=line_boundary_start, end=end,
                init=boundary_init,
                compute_base_twiss=compute_base_twiss)
            completed_init = first_table.completed_init
        else:
            second_table = _compute_twiss_range_piece(
                kwargs, start=line_boundary_start, end=end, init=init,
                compute_base_twiss=compute_base_twiss)
            boundary_init = second_table.get_twiss_init(line_boundary_start)
            boundary_init.element_name = line_boundary_end
            first_table = _compute_twiss_range_piece(
                kwargs, start=start, end=line_boundary_end,
                init=boundary_init,
                compute_base_twiss=compute_base_twiss)
            completed_init = second_table.completed_init

        twiss_tables = (first_table, second_table)

    # Assemble the output in traversal order and align it with the supplied init.
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


def _compute_twiss_range_piece(
        data, *, start, end, init, compute_base_twiss):

    piece_data = data.copy()
    piece_data.update(
        start=start,
        end=end,
        init=init,
        completed_init=init.copy(),
    )
    return compute_base_twiss(piece_data)


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
