# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from dataclasses import dataclass

from .element_indexing import _str_to_index


@dataclass(frozen=True)
class _TwissPropagationRequest:
    line: object
    start: object
    end: object
    reverse: bool
    periodic: bool
    periodic_mode: object
    init_at: object

    @property
    def requested_direction(self):
        return 'backward' if self.reverse else 'forward'


@dataclass(frozen=True)
class _TwissInitAcquisitionPlan:
    source: str
    scope: str
    computation_direction: str
    init_at: object


@dataclass(frozen=True)
class _OpenTwissPiecePlan:
    role: str
    start: object
    end: object
    init_at: object


@dataclass(frozen=True)
class _LoopAroundTwissPlan:
    first_table_piece: object
    second_table_piece: object
    init_piece_role: str
    transfer_init_at: object
    transfer_init_element_name: object


@dataclass(frozen=True)
class _OpenOneTurnTwissPlan:
    first_piece: object
    second_piece: object
    transfer_init_at: object
    transfer_init_element_name: object


@dataclass(frozen=True)
class _OpenTwissPropagationPlan:
    output_direction: str
    crosses_line_start: bool
    init_is_at_boundary: bool
    pieces: tuple


@dataclass(frozen=True)
class _TwissComputationPlan:
    init_acquisition: object
    open_propagation: object


def _make_twiss_propagation_request(kwargs):

    return _TwissPropagationRequest(
        line=kwargs['line'],
        start=kwargs['start'],
        end=kwargs['end'],
        reverse=kwargs['reverse'],
        periodic=kwargs['periodic'],
        periodic_mode=kwargs['periodic_mode'],
        init_at=kwargs['init_at'],
    )


def _plan_twiss_computation(kwargs, init):
    """Describe the target non-recursive Twiss orchestration.

    Init acquisition and Twiss propagation are separate phases. Periodic Twiss
    first obtains a TwissInit, then propagation is handled by the same open-Twiss
    segment planner used by explicitly open Twiss requests.
    For reverse requests, start and end are already in the requested traversal
    order. Periodic reverse requests should still prefer a forward periodic
    solution and only reverse/order the table output afterward where possible.
    """

    request = _make_twiss_propagation_request(kwargs)
    init_acquisition = _plan_twiss_init_acquisition(request)
    init_element_name = _planned_open_twiss_init_element_name(request, init)
    open_propagation = _plan_open_twiss_propagation(
        request=request, init_element_name=init_element_name)

    return _TwissComputationPlan(
        init_acquisition=init_acquisition,
        open_propagation=open_propagation,
    )


def _plan_twiss_init_acquisition(request):

    if request.periodic:
        source = 'periodic_solution'
        computation_direction = 'forward'
        if request.start is None and request.end is None:
            scope = 'full_line'
        else:
            scope = 'requested_range'
    else:
        source = 'open_input'
        scope = 'not_applicable'
        computation_direction = request.requested_direction

    return _TwissInitAcquisitionPlan(
        source=source,
        scope=scope,
        computation_direction=computation_direction,
        init_at=request.init_at,
    )


def _planned_open_twiss_init_element_name(request, init):

    if hasattr(init, 'element_name'):
        return init.element_name
    if request.init_at is not None:
        return request.init_at
    if request.start is not None:
        return request.start
    return request.line._element_names_unique[0]


def _plan_open_twiss_propagation(request, init_element_name):
    """Plan open Twiss propagation after a TwissInit is available.

    The plan is expressed as output-table pieces. Active composed Twiss routes
    map these pieces to segment calls and keep table-combination policy in the
    route-specific executors.
    """

    crosses_line_start = _twiss_request_crosses_line_start(request)
    init_is_at_boundary = init_element_name in (request.start, request.end)

    if init_is_at_boundary and not crosses_line_start:
        pieces = (_OpenTwissPiecePlan(
            role='boundary_init',
            start=request.start,
            end=request.end,
            init_at=init_element_name,
        ),)
    else:
        pieces = _plan_split_open_twiss_pieces(
            request=request,
            init_element_name=init_element_name,
            crosses_line_start=crosses_line_start,
        )

    return _OpenTwissPropagationPlan(
        output_direction=request.requested_direction,
        crosses_line_start=crosses_line_start,
        init_is_at_boundary=init_is_at_boundary,
        pieces=pieces,
    )


def _assert_open_plan_for_init_inside_range(plan):

    assert not plan.crosses_line_start
    assert not plan.init_is_at_boundary
    assert [piece.role for piece in plan.pieces] == [
        'before_init', 'after_init']


def _assert_open_plan_for_loop_around(plan):

    assert plan.crosses_line_start
    assert len(plan.pieces) in (2, 3)


def _twiss_request_crosses_line_start(request):

    if request.start is None or request.end is None:
        return False

    rv = -1 if request.reverse else 1
    return (
        rv * _str_to_index(request.line, request.start)
        > rv * _str_to_index(request.line, request.end)
    )


def _plan_split_open_twiss_pieces(request, init_element_name,
                                  crosses_line_start):

    pieces = []
    if crosses_line_start:
        pieces.extend(_plan_line_start_split_pieces(request, init_element_name))
    else:
        pieces.extend(_plan_init_split_pieces(request, init_element_name))

    return tuple(pieces)


def _plan_init_split_pieces(request, init_element_name):

    return (
        _OpenTwissPiecePlan(
            role='before_init',
            start=request.start,
            end=init_element_name,
            init_at=init_element_name,
        ),
        _OpenTwissPiecePlan(
            role='after_init',
            start=init_element_name,
            end=request.end,
            init_at=init_element_name,
        ),
    )


def _plan_line_start_split_pieces(request, init_element_name):

    line_start = request.line._element_names_unique[0]
    line_end = request.line._element_names_unique[-1]
    line_boundary_end = line_end if not request.reverse else line_start
    line_boundary_start = line_start if not request.reverse else line_end

    if init_element_name in (request.start, request.end):
        return (
            _OpenTwissPiecePlan(
                role='start_to_line_boundary',
                start=request.start,
                end=line_boundary_end,
                init_at=init_element_name,
            ),
            _OpenTwissPiecePlan(
                role='line_boundary_to_end',
                start=line_boundary_start,
                end=request.end,
                init_at=init_element_name,
            ),
        )

    init_index = _str_to_index(request.line, init_element_name)
    start_index = _str_to_index(request.line, request.start)

    if ((not request.reverse and init_index >= start_index)
            or (request.reverse and init_index <= start_index)):
        return (
            _OpenTwissPiecePlan(
                role='start_to_init',
                start=request.start,
                end=init_element_name,
                init_at=init_element_name,
            ),
            _OpenTwissPiecePlan(
                role='init_to_line_boundary',
                start=init_element_name,
                end=line_boundary_end,
                init_at=init_element_name,
            ),
            _OpenTwissPiecePlan(
                role='line_boundary_to_end',
                start=line_boundary_start,
                end=request.end,
                init_at=init_element_name,
            ),
        )

    return (
        _OpenTwissPiecePlan(
            role='start_to_line_boundary',
            start=request.start,
            end=line_boundary_end,
            init_at=init_element_name,
        ),
        _OpenTwissPiecePlan(
            role='line_boundary_to_init',
            start=line_boundary_start,
            end=init_element_name,
            init_at=init_element_name,
        ),
        _OpenTwissPiecePlan(
            role='init_to_end',
            start=init_element_name,
            end=request.end,
            init_at=init_element_name,
        ),
    )


def _plan_loop_around_twiss_parts(line, start, end, init, reverse):

    ele_name_init = init.element_name
    if not reverse:
        assert _str_to_index(line, end) < _str_to_index(line, start), (
            'This function should not have been called')
    else:
        assert _str_to_index(line, end) > _str_to_index(line, start), (
            'This function should not have been called')

    request = _TwissPropagationRequest(
        line=line,
        start=start,
        end=end,
        reverse=reverse,
        periodic=False,
        periodic_mode=None,
        init_at=ele_name_init,
    )
    open_plan = _plan_open_twiss_propagation(
        request=request, init_element_name=ele_name_init)

    _assert_open_plan_for_loop_around(open_plan)
    first_table_piece, second_table_piece = (
        _loop_around_table_pieces_from_open_plan(open_plan))
    init_piece_role = _loop_around_init_piece_role(
        line=line, start=start, end=end, init=init, reverse=reverse)
    if init_piece_role == 'first_table_piece':
        transfer_init_at = '_end_point'
        transfer_init_element_name = second_table_piece.start
    else:
        transfer_init_at = second_table_piece.start
        transfer_init_element_name = first_table_piece.end

    return _LoopAroundTwissPlan(
        first_table_piece=first_table_piece,
        second_table_piece=second_table_piece,
        init_piece_role=init_piece_role,
        transfer_init_at=transfer_init_at,
        transfer_init_element_name=transfer_init_element_name,
    )


def _loop_around_table_pieces_from_open_plan(open_plan):

    if len(open_plan.pieces) == 2:
        return open_plan.pieces

    if len(open_plan.pieces) != 3:
        raise RuntimeError('Unexpected loop-around Twiss plan')

    if open_plan.pieces[0].role == 'start_to_init':
        first_piece = _join_loop_around_piece_pair(
            open_plan.pieces[0], open_plan.pieces[1])
        second_piece = open_plan.pieces[2]
        return first_piece, second_piece

    if open_plan.pieces[0].role == 'start_to_line_boundary':
        first_piece = open_plan.pieces[0]
        second_piece = _join_loop_around_piece_pair(
            open_plan.pieces[1], open_plan.pieces[2])
        return first_piece, second_piece

    raise RuntimeError('Unexpected loop-around Twiss plan')


def _join_loop_around_piece_pair(first_piece, second_piece):

    return _OpenTwissPiecePlan(
        role=f'{first_piece.role}+{second_piece.role}',
        start=first_piece.start,
        end=second_piece.end,
        init_at=first_piece.init_at,
    )


def _loop_around_init_piece_role(line, start, end, init, reverse):

    init_index = _str_to_index(line, init.element_name)
    start_index = _str_to_index(line, start)
    end_index = _str_to_index(line, end)

    if not reverse:
        if init_index >= start_index:
            return 'first_table_piece'
        if init_index <= end_index:
            return 'second_table_piece'
    else:
        if init_index <= start_index:
            return 'first_table_piece'
        if init_index >= end_index:
            return 'second_table_piece'

    raise RuntimeError(
        'Boundary conditions not at start or end of the specified range')


def _plan_init_inside_range_twiss_parts(line, start, end, init, reverse):

    request = _TwissPropagationRequest(
        line=line,
        start=start,
        end=end,
        reverse=reverse,
        periodic=False,
        periodic_mode=None,
        init_at=init.element_name,
    )
    plan = _plan_open_twiss_propagation(
        request=request, init_element_name=init.element_name)

    _assert_open_plan_for_init_inside_range(plan)

    return plan


def _plan_open_one_turn_twiss(line, start, reverse):

    if reverse:
        line_boundary_end = line._element_names_unique[0]
        line_boundary_start = line._element_names_unique[-1]
    else:
        line_boundary_end = line._element_names_unique[-1]
        line_boundary_start = line._element_names_unique[0]

    first_piece = _OpenTwissPiecePlan(
        role='start_to_line_boundary',
        start=start,
        end=line_boundary_end,
        init_at=start,
    )
    second_piece = _OpenTwissPiecePlan(
        role='line_boundary_to_start',
        start=line_boundary_start,
        end=start,
        init_at=line_boundary_start,
    )

    return _OpenOneTurnTwissPlan(
        first_piece=first_piece,
        second_piece=second_piece,
        transfer_init_at='_end_point',
        transfer_init_element_name=line_boundary_start,
    )
