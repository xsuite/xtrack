"""Anchor semantics and longitudinal coordinate resolution."""

import numpy as np
import xdeps as xd

from .models import ResolvedPlacement


_ALLOWED_ANCHORS = (None, 'center', 'centre', 'start', 'end')


def _resolve_s_positions(seq_all_places, env, refer='center', diagnostics=False):
    """Resolve placement specifications to a table of absolute coordinates."""
    places = list(seq_all_places)

    # Check that relative placement arguments are used consistently.
    for place in places:
        if place.from_anchor is not None and place.from_ is None:
            raise ValueError(
                'Cannot specify `from_anchor` without providing `from_`. '
                f'Error in placement `{place}`.'
            )
        if place.from_ is not None and place.at is None:
            raise ValueError(
                'Cannot specify `from_` without providing `at`. '
                f'Error in placement `{place}`.'
            )

    # Use an auxiliary line to build a table of all the elements (specified order)
    aux_line = env.new_line(
        components=[place.name for place in places],
        refer=refer,
    )
    table = aux_line.get_table()
    table['length'] = np.diff(table.s, append=table.s[-1])
    table = table.rows[:-1]

    # If places is empty, return an empty table with the correct columns.
    if not places:
        table['s_start'] = np.array([], dtype=float)
        table['s_center'] = np.array([], dtype=float)
        table['s_end'] = np.array([], dtype=float)
        table['s'] = np.array([], dtype=float)
        table['from_'] = np.array([], dtype=object)
        table['from_anchor'] = np.array([], dtype=object)
        return table

    # Build a dictionary of element lengths by name for quick lookup.
    length_by_name = dict(zip(table.env_name, table.length))

    # Track resolved placements by input order and by referenced component name.
    resolved_by_index = {}
    resolved_by_name = {}

    # If first place has at=None, it goes at s=0
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

    # Scan all unresolved places and resolve those whose reference component
    # has already been resolved. Repeat the scan to handle forward references
    # to components appearing later in the input. Stop when all places are
    # resolved or when a full scan makes no progress.
    made_progress = True
    while made_progress:
        made_progress = False
        for index, place in enumerate(places):
            if index in resolved_by_index:
                continue

            from_name = None
            from_anchor = None

            self_length = length_by_name[place.name]

            if place.at is None:
                # Needs to be placed right after the previous
                # component in the input list.

                # Check if the previous component has been resolved.
                previous = resolved_by_index.get(index - 1, None)
                if previous is None:
                    continue  # Cannot resolve this place yet

                s_start = previous.s_end

                if not str(previous.name).startswith('||drift'):
                    # we keep track of the element to which it is referred
                    # to handle sandwiches of thin elements.
                    from_name = previous.name
                    from_anchor = 'end'

            elif place.from_ is None:
                # Needs to be placed at an absolute location along the line.
                # Needs to be placed based on `at`, `from_`, and `from_anchor`.

                at = _evaluate_position_expression(place.at, aux_line._xdeps_eval)

                # Component anchor (start/end/center)
                anchor = refer if place.anchor is None else place.anchor

                # Absolute location of the component
                s_start = at - _anchor_offset(anchor, self_length)

            else:
                # Needs to be placed relative to another component,
                # based on `at`, `from_`, and `from_anchor`.

                at = _evaluate_position_expression(place.at, aux_line._xdeps_eval)

                # Check if the referenced component has been resolved.
                reference = resolved_by_name.get(place.from_)
                if reference is None:
                    continue  # Cannot resolve this place yet

                # Identify reference anchor (start/end/center)
                if place.from_anchor is not None:
                    reference_anchor = place.from_anchor
                else:
                    reference_anchor = refer

                # Absolute s coordinate of the reference anchor point
                s_reference = reference.s_start + _anchor_offset(
                    reference_anchor, reference.length
                )

                # Component anchor (start/end/center)
                anchor = refer if place.anchor is None else place.anchor

                # Absolute location of the component
                s_start = s_reference + at - _anchor_offset(anchor, self_length)
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


def _evaluate_position_expression(at, evaluator):
    if isinstance(at, str):
        at = evaluator.eval(at)
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
