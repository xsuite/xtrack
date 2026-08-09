"""Anchor semantics and longitudinal coordinate resolution."""

import numpy as np
import xdeps as xd
import xtrack as xt


_ALLOWED_ANCHORS = (None, 'center', 'centre', 'start', 'end')


def _anchor_offset(anchor, length):
    """Return the distance from an element start to the selected anchor."""
    if anchor in ('center', 'centre'):
        return length / 2
    if anchor == 'end':
        return length
    if anchor == 'start':
        return 0
    raise ValueError(f'Unknown anchor {anchor!r}.')


def _with_default_anchor(anchor, default_anchor):
    return default_anchor if anchor is None else anchor


def _resolve_one_position(
    at, anchor, from_anchor, self_length, from_length, s_start_from, default_anchor
):
    """Resolve one absolute start coordinate from an anchor relationship."""
    if xd.refs.is_ref(at):
        at = at._value

    anchor = _with_default_anchor(anchor, default_anchor)
    from_anchor = _with_default_anchor(from_anchor, default_anchor)

    s_from = 0
    if from_length is not None:
        s_from = s_start_from + _anchor_offset(from_anchor, from_length)
    return s_from + at - _anchor_offset(anchor, self_length)


def _prepare_position_table(places, env, refer):
    """Collect element lengths and create the result table skeleton."""
    names = [place.name for place in places]
    aux_line = env.new_line(components=names, refer=refer)
    table = aux_line.get_table()
    table['length'] = np.diff(table.s, append=table.s[-1])
    table = table.rows[:-1]
    lengths = xt.Table({'name': table.env_name, 'length': table.length})
    return aux_line, table, lengths


def _add_empty_position_columns(table):
    table['s_start'] = np.array([], dtype=float)
    table['s_center'] = np.array([], dtype=float)
    table['s_end'] = np.array([], dtype=float)
    table['s'] = np.array([], dtype=float)
    table['from_'] = np.array([], dtype=object)
    table['from_anchor'] = np.array([], dtype=object)
    return table


def _evaluate_position_expression(at, evaluator):
    if isinstance(at, str):
        return evaluator.eval(at)
    if xd.refs.is_ref(at):
        return at._value
    return at


def _try_resolve_place(
    index,
    places,
    start_by_place,
    place_by_name,
    lengths,
    evaluator,
    refer,
):
    """Resolve one place if all of its dependencies are available."""
    place = places[index]
    if place in start_by_place:
        return False

    if (place.from_ is not None or place.from_anchor is not None) and place.at is None:
        raise ValueError(
            'Cannot specify `from_` or `from_anchor` without providing `at`. '
            f'Error in place `{place}`.'
        )

    if place.at is None:
        previous = places[index - 1]
        if previous not in start_by_place:
            return False
        start_by_place[place] = (
            start_by_place[previous] + lengths['length', previous.name]
        )
        place_by_name[place.name] = place
        if not previous.name.startswith('||drift'):
            place.at = 0
            place.from_ = previous.name
            place.from_anchor = 'end'
        return True

    at = _evaluate_position_expression(place.at, evaluator)
    from_length = None
    s_start_from = None
    if place.from_ is not None:
        if place.from_ not in place_by_name:
            return False
        from_place = place_by_name[place.from_]
        from_length = lengths['length', place.from_]
        s_start_from = start_by_place[from_place]

    start_by_place[place] = _resolve_one_position(
        at,
        anchor=place.anchor,
        from_anchor=place.from_anchor,
        self_length=lengths['length', place.name],
        from_length=from_length,
        s_start_from=s_start_from,
        default_anchor=refer,
    )
    place_by_name[place.name] = place
    return True


def _resolve_place_coordinates(places, lengths, evaluator, refer):
    """Resolve all placements, iterating until no dependency can progress."""
    start_by_place = {}
    place_by_name = {}

    if places[0].at is None:
        start_by_place[places[0]] = 0
        place_by_name[places[0].name] = places[0]

    made_progress = True
    while made_progress:
        made_progress = False
        for index in range(len(places)):
            if _try_resolve_place(
                index,
                places,
                start_by_place,
                place_by_name,
                lengths,
                evaluator,
                refer,
            ):
                made_progress = True

    if len(start_by_place) != len(places):
        unresolved = set(places) - set(start_by_place)
        raise ValueError(f'Could not resolve all s positions: {unresolved}')
    return start_by_place


def _add_resolved_position_columns(table, places, start_by_place):
    starts = np.array([start_by_place[place] for place in places])
    table['s_start'] = starts
    table['s_center'] = starts + table['length'] / 2
    table['s_end'] = starts + table['length']
    table['s'] = starts
    table['from_'] = np.array([place.from_ for place in places])
    table['from_anchor'] = np.array([place.from_anchor for place in places])
    return table


def _resolve_s_positions(seq_all_places, env, refer='center'):
    """Resolve placement specifications to a table of absolute coordinates."""
    places = [place.copy() for place in seq_all_places]
    aux_line, table, lengths = _prepare_position_table(places, env, refer)
    if not places:
        return _add_empty_position_columns(table)

    start_by_place = _resolve_place_coordinates(
        places, lengths, aux_line._xdeps_eval, refer
    )
    return _add_resolved_position_columns(table, places, start_by_place)
