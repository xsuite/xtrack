# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from dataclasses import dataclass

from .base_open_propagation import (
    _TwissPropagationRequest,
    _plan_open_one_turn_twiss,
    _plan_open_twiss_propagation,
)


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
    route = _plan_twiss_route(request=request, init=init)
    init_acquisition = _plan_twiss_init_acquisition(
        request=request, route=route)
    one_turn_propagation = _plan_one_turn_twiss_propagation(
        request=request, route=route)
    init_element_name = _planned_open_twiss_init_element_name(request, init)
    open_propagation = _plan_open_twiss_propagation(
        request=request, init_element_name=init_element_name)

    return _TwissComputationPlan(
        route=route,
        init_acquisition=init_acquisition,
        one_turn_propagation=one_turn_propagation,
        open_propagation=open_propagation,
    )


def _plan_twiss_route(request, init):

    if request.start is not None and request.end is None:
        if request.periodic:
            return 'periodic_one_turn_from_start'
        return 'open_one_turn_from_start'

    if (init == 'full_periodic'
            and (request.start is not None or request.end is not None)):
        return 'full_periodic_range'

    return 'base'


def _plan_twiss_init_acquisition(request, route):

    if route == 'full_periodic_range':
        source = 'full_periodic_solution'
        scope = 'full_line'
        computation_direction = 'forward'
    elif route == 'periodic_one_turn_from_start':
        source = 'periodic_solution'
        scope = 'full_line'
        computation_direction = 'forward'
    elif request.periodic:
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


def _plan_one_turn_twiss_propagation(request, route):

    if route == 'periodic_one_turn_from_start':
        return _PeriodicOneTurnTwissPlan(
            start=request.start,
            output_direction=request.requested_direction,
        )

    if route == 'open_one_turn_from_start':
        return _plan_open_one_turn_twiss(
            line=request.line,
            start=request.start,
            reverse=request.reverse,
        )

    return None


def _planned_open_twiss_init_element_name(request, init):

    if request.init_at is not None:
        return request.init_at
    if hasattr(init, 'element_name') and init.element_name is not None:
        return init.element_name
    if request.start is not None:
        return request.start
    return request.line._element_names_unique[0]
