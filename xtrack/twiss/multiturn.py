# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import xtrack as xt  # To avoid circular imports


def _kwargs_for_multiturn_continuation(kwargs, data):
    """Refresh public-call kwargs for recursive multi-turn continuation."""

    out = kwargs.copy()
    for key in kwargs:
        if key in data:
            out[key] = data[key]
    # Apply zero_at only once, after the complete multi-turn table is assembled.
    out['zero_at'] = None
    out.pop('input_kwargs', None)
    return out


def _extend_twiss_result_to_multiple_turns(twiss_res, num_turns, kwargs):

    kwargs.pop('num_turns')
    kwargs.pop('init')
    kwargs.pop('start')
    kwargs.pop('end')

    tw_mt = _multiturn_twiss(
        tw0=twiss_res, num_turns=num_turns, kwargs=kwargs)
    tw_mt._data['_tw0'] = twiss_res
    return tw_mt


def _multiturn_twiss(tw0, num_turns, kwargs):

    twisses_to_merge = _compute_multiturn_twiss_parts(
        tw0=tw0, num_turns=num_turns, kwargs=kwargs)
    return _combine_multiturn_twiss_tables(twisses_to_merge)


def _compute_multiturn_twiss_parts(tw0, num_turns, kwargs):

    tw_curr = tw0
    twisses_to_merge = []

    for i_turn in range(num_turns):
        twisses_to_merge.append(
            _multiturn_start_row(tw_curr=tw_curr, i_turn=i_turn))
        twisses_to_merge.append(tw_curr)

        if i_turn == num_turns - 1:
            break  # need n-1 twisses

        tw_curr = _continue_multiturn_twiss(tw_curr=tw_curr, kwargs=kwargs)

    return twisses_to_merge


def _multiturn_start_row(tw_curr, i_turn):

    tw_start_turn = tw_curr.rows[0]
    tw_start_turn.name[0] = f'_turn_{i_turn}'
    return tw_start_turn


def _continue_multiturn_twiss(tw_curr, kwargs):

    # This local import is the one intentional recursive dependency.
    from .twiss import twiss_line

    line = kwargs['line']
    tini1 = tw_curr.get_twiss_init(-1)
    tini1.element_name = tw_curr.name[0]

    return twiss_line(
        **kwargs, init=tini1, start=tw_curr.name[0],
        end=line._element_names_unique[-1])


def _combine_multiturn_twiss_tables(twisses_to_merge):

    return xt.TwissTable.concatenate(twisses_to_merge)
