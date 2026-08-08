# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from .constants import VARS_FOR_TWISS_INIT_GENERATION
from .segment_computation import (
    _compute_twiss_segment,
    _compute_twiss_segment_for_piece,
)

import xtrack as xt  # To avoid circular imports


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

