# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from dataclasses import dataclass

from .open_propagation import (
    _TwissPropagationRequest,
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
