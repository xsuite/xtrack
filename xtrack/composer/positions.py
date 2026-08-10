"""Anchor semantics and longitudinal coordinate resolution."""

import numpy as np
import xdeps as xd

from .models import ResolvedPlacement


_ALLOWED_ANCHORS = (None, 'center', 'centre', 'start', 'end')


def _resolve_s_positions(seq_all_places, env, refer='center', diagnostics=False):
    """Resolve placement specifications to a table of absolute coordinates."""
    places = list(seq_all_places)

    # Use an auxiliary line to build a table of all the elements
    aux_line = env.new_line(
        components=[place.name for place in places],
        refer=refer,
    )
    table = aux_line.get_table()
    table['length'] = np.diff(table.s, append=table.s[-1])
    table = table.rows[:-1]

    if not places:
        table['s_start'] = np.array([], dtype=float)
        table['s_center'] = np.array([], dtype=float)
        table['s_end'] = np.array([], dtype=float)
        table['s'] = np.array([], dtype=float)
        table['from_'] = np.array([], dtype=object)
        table['from_anchor'] = np.array([], dtype=object)
        return table

    length_by_name = dict(zip(table.env_name, table.length))

    # Track resolved placements by input order and by referenced component name.
    resolved_by_index = {}
    resolved_by_name = {}

    if places[0].at is None:
        place = places[0]
        resolved = ResolvedPlacement(
            source_index=0,
            name=place.name,
            table_name=str(table.name[0]),
            env_name=str(table.env_name[0]),
            length=length_by_name[place.name],
            isthick=bool(table.isthick[0]),
            s_start=0,
            from_=place.from_,
            from_anchor=place.from_anchor,
        )
        resolved_by_index[0] = resolved
        resolved_by_name[place.name] = resolved

    made_progress = True
    while made_progress:
        made_progress = False
        for index, place in enumerate(places):
            if index in resolved_by_index:
                continue

            if (
                place.from_ is not None or place.from_anchor is not None
            ) and place.at is None:
                raise ValueError(
                    'Cannot specify `from_` or `from_anchor` without providing '
                    f'`at`. Error in placement `{place}`.'
                )

            self_length = length_by_name[place.name]
            if place.at is None:
                previous = resolved_by_index.get(index - 1)
                if previous is None:
                    continue
                from_name = place.from_
                from_anchor = place.from_anchor
                if not str(previous.name).startswith('||drift'):
                    from_name = previous.name
                    from_anchor = 'end'
                s_start = previous.s_end
            else:
                at = _evaluate_position_expression(place.at, aux_line._xdeps_eval)
                from_length = None
                s_start_from = None
                if place.from_ is not None:
                    reference = resolved_by_name.get(place.from_)
                    if reference is None:
                        continue
                    from_length = reference.length
                    s_start_from = reference.s_start

                s_start = _resolve_one_position(
                    at,
                    anchor=place.anchor,
                    from_anchor=place.from_anchor,
                    self_length=self_length,
                    from_length=from_length,
                    s_start_from=s_start_from,
                    default_anchor=refer,
                )
                from_name = place.from_
                from_anchor = place.from_anchor

            resolved = ResolvedPlacement(
                source_index=index,
                name=place.name,
                table_name=str(table.name[index]),
                env_name=str(table.env_name[index]),
                length=self_length,
                isthick=bool(table.isthick[index]),
                s_start=s_start,
                from_=from_name,
                from_anchor=from_anchor,
            )
            resolved_by_index[index] = resolved
            resolved_by_name[place.name] = resolved
            made_progress = True

    if len(resolved_by_index) != len(places):
        if not diagnostics:
            raise ValueError(
                'Could not resolve all placement positions. Call '
                'Composer.validate() or enable diagnostics for details.'
            )
        unresolved_indices = [
            index for index in range(len(places)) if index not in resolved_by_index
        ]
        _raise_resolution_error(places, unresolved_indices)

    placements = [resolved_by_index[index] for index in range(len(places))]
    table['s_start'] = np.array([place.s_start for place in placements])
    table['s_center'] = np.array([place.s_center for place in placements])
    table['s_end'] = np.array([place.s_end for place in placements])
    table['s'] = table['s_start'].copy()
    table['from_'] = np.array([place.from_ for place in placements])
    table['from_anchor'] = np.array([place.from_anchor for place in placements])
    return table


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


def _evaluate_position_expression(at, evaluator):
    if isinstance(at, str):
        return evaluator.eval(at)
    if xd.refs.is_ref(at):
        return at._value
    return at


def _format_place(index, place):
    return f'component {index} ({place.name!r})'


def _find_dependency_cycle(places, unresolved_indices):
    """Return one cycle of component indices, including its repeated endpoint."""
    unresolved_indices = set(unresolved_indices)
    indices_by_name = {}
    for index, place in enumerate(places):
        indices_by_name.setdefault(place.name, []).append(index)

    dependency_by_index = {}
    for index in unresolved_indices:
        place = places[index]
        dependency = None
        if place.from_ is not None:
            candidates = indices_by_name.get(place.from_, [])
            dependency = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate in unresolved_indices
                ),
                None,
            )
        elif place.at is None and index > 0:
            dependency = index - 1
        if dependency in unresolved_indices:
            dependency_by_index[index] = dependency

    for start in dependency_by_index:
        path = []
        path_index = {}
        current = start
        while current in dependency_by_index:
            if current in path_index:
                cycle = path[path_index[current] :]
                return cycle + [current]
            path_index[current] = len(path)
            path.append(current)
            current = dependency_by_index[current]
    return None


def _raise_resolution_error(places, unresolved_indices):
    """Raise a specific diagnostic for a stalled dependency resolution."""
    available_names = {place.name for place in places}
    missing = [
        index
        for index in unresolved_indices
        if places[index].from_ is not None
        and places[index].from_ not in available_names
    ]
    if missing:
        details = '; '.join(
            f'{_format_place(index, places[index])} references missing element '
            f'{places[index].from_!r}'
            for index in missing
        )
        blocked = ', '.join(
            _format_place(index, places[index]) for index in unresolved_indices
        )
        raise ValueError(f'Missing placement reference: {details}. Blocked: {blocked}.')

    cycle = _find_dependency_cycle(places, unresolved_indices)
    if cycle is not None:
        chain = ' -> '.join(_format_place(index, places[index]) for index in cycle)
        raise ValueError(f'Cyclic placement dependency: {chain}.')

    blocked = ', '.join(
        _format_place(index, places[index]) for index in unresolved_indices
    )
    raise ValueError(f'Could not resolve placement dependencies: {blocked}.')
