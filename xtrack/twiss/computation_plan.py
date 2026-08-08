# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from dataclasses import dataclass

from .element_indexing import _str_to_index


@dataclass(frozen=True)
class _TwissInitAcquisitionPlan:
    source: str
    scope: str
    computation_direction: str
    init_at: object


@dataclass(frozen=True)
class _TwissComputationPlan:
    route: str
    init_acquisition: object
    one_turn_propagation: object
    open_propagation: object


@dataclass(frozen=True)
class _PeriodicOneTurnTwissPlan:
    start: object
    output_direction: str


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
        # With reverse=True, start/end are already in reverse traversal order.
        return 'backward' if self.reverse else 'forward'


@dataclass(frozen=True)
class _OpenTwissPiecePlan:
    role: str
    start: object
    end: object
    init_at: object


@dataclass(frozen=True)
class _OpenOneTurnTwissPlan:
    first_piece: object
    second_piece: object
    transfer_init_at: object
    transfer_init_element_name: object


@dataclass(frozen=True)
class _OpenTwissPropagationPlan:
    output_direction: str
    crosses_line_boundary: bool
    init_is_at_boundary: bool
    pieces: tuple


def _plan_twiss_computation(kwargs, init):
    """Describe init acquisition and non-recursive Twiss propagation."""

    request = _TwissPropagationRequest(
        line=kwargs['line'],
        start=kwargs['start'],
        end=kwargs['end'],
        reverse=kwargs['reverse'],
        periodic=kwargs['periodic'],
        periodic_mode=kwargs['periodic_mode'],
        init_at=kwargs['init_at'],
    )

    if request.start is not None and request.end is None:
        if request.periodic:
            route = 'periodic_one_turn_from_start'
        else:
            route = 'open_one_turn_from_start'
    elif (init == 'full_periodic'
            and (request.start is not None or request.end is not None)):
        route = 'full_periodic_range'
    else:
        route = 'base'

    if route == 'full_periodic_range':
        init_acquisition = _TwissInitAcquisitionPlan(
            source='full_periodic_solution',
            scope='full_line',
            computation_direction='forward',
            init_at=request.init_at,
        )
    elif route == 'periodic_one_turn_from_start':
        init_acquisition = _TwissInitAcquisitionPlan(
            source='periodic_solution',
            scope='full_line',
            computation_direction='forward',
            init_at=request.init_at,
        )
    elif request.periodic:
        if request.start is None and request.end is None:
            periodic_scope = 'full_line'
        else:
            periodic_scope = 'requested_range'
        init_acquisition = _TwissInitAcquisitionPlan(
            source='periodic_solution',
            scope=periodic_scope,
            computation_direction='forward',
            init_at=request.init_at,
        )
    else:
        init_acquisition = _TwissInitAcquisitionPlan(
            source='open_input',
            scope='not_applicable',
            computation_direction=request.requested_direction,
            init_at=request.init_at,
        )

    if route == 'periodic_one_turn_from_start':
        one_turn_propagation = _PeriodicOneTurnTwissPlan(
            start=request.start,
            output_direction=request.requested_direction,
        )
    elif route == 'open_one_turn_from_start':
        if request.reverse:
            line_boundary_end = request.line._element_names_unique[0]
            line_boundary_start = request.line._element_names_unique[-1]
        else:
            line_boundary_end = request.line._element_names_unique[-1]
            line_boundary_start = request.line._element_names_unique[0]

        one_turn_propagation = _OpenOneTurnTwissPlan(
            first_piece=_OpenTwissPiecePlan(
                role='start_to_line_boundary',
                start=request.start,
                end=line_boundary_end,
                init_at=request.start,
            ),
            second_piece=_OpenTwissPiecePlan(
                role='line_boundary_to_start',
                start=line_boundary_start,
                end=request.start,
                init_at=line_boundary_start,
            ),
            transfer_init_at='_end_point',
            transfer_init_element_name=line_boundary_start,
        )
    else:
        one_turn_propagation = None

    if request.init_at is not None:
        init_element_name = request.init_at
    elif hasattr(init, 'element_name') and init.element_name is not None:
        init_element_name = init.element_name
    elif request.start is not None:
        init_element_name = request.start
    else:
        init_element_name = request.line._element_names_unique[0]

    if request.start is None or request.end is None:
        crosses_line_boundary = False
    else:
        direction_sign = -1 if request.reverse else 1
        crosses_line_boundary = (
            direction_sign * _str_to_index(request.line, request.start)
            > direction_sign * _str_to_index(request.line, request.end)
        )

    init_is_at_boundary = init_element_name in (request.start, request.end)
    if init_is_at_boundary and not crosses_line_boundary:
        open_pieces = (_OpenTwissPiecePlan(
            role='boundary_init',
            start=request.start,
            end=request.end,
            init_at=init_element_name,
        ),)
    else:
        open_pieces = _plan_open_twiss_pieces(
            request=request,
            init_element_name=init_element_name,
            crosses_line_boundary=crosses_line_boundary,
        )

    return _TwissComputationPlan(
        route=route,
        init_acquisition=init_acquisition,
        one_turn_propagation=one_turn_propagation,
        open_propagation=_OpenTwissPropagationPlan(
            output_direction=request.requested_direction,
            crosses_line_boundary=crosses_line_boundary,
            init_is_at_boundary=init_is_at_boundary,
            pieces=open_pieces,
        ),
    )


def _plan_open_twiss_pieces(
        request, init_element_name, crosses_line_boundary):

    if not crosses_line_boundary:
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
    init_is_after_start = (
        (not request.reverse and init_index >= start_index)
        or (request.reverse and init_index <= start_index)
    )
    if init_is_after_start:
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
