"""Anchor semantics and longitudinal coordinate resolution."""

import numpy as np
import xdeps as xd


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

    # Track resolved positions by input order and referenced component name.
    s_start_by_index = [None] * len(places)
    from_name_by_index = [None] * len(places)
    from_anchor_by_index = [None] * len(places)
    resolved_index_by_name = {}

    # If first place has at=None, it goes at s=0
    if places[0].at is None:
        s_start_by_index[0] = 0
        resolved_index_by_name[places[0].name] = 0

    # Scan all unresolved places and resolve those whose reference component
    # has already been resolved. Repeat the scan to handle forward references
    # to components appearing later in the input. Stop when all places are
    # resolved or when a full scan makes no progress.
    made_progress = True
    while made_progress:
        made_progress = False
        for index, place in enumerate(places):
            if s_start_by_index[index] is not None:
                continue  # Already resolved

            from_name = None
            from_anchor = None
            self_length = length_by_name[place.name]

            if place.at is None:
                # Needs to be placed right after the previous
                # component in the input list.

                # Check if the previous component has been resolved.
                previous_index = index - 1
                previous_s_start = s_start_by_index[previous_index]
                if previous_s_start is None:
                    continue  # Cannot resolve this place yet

                previous = places[previous_index]
                s_start = previous_s_start + length_by_name[previous.name]
                if not str(previous.name).startswith('||drift'):
                    # we keep track of the element to which it is referred
                    # to handle sandwiches of thin elements.
                    from_name = str(table.name[previous_index])
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
                reference_index = resolved_index_by_name.get(place.from_)
                if reference_index is None:
                    continue  # Cannot resolve this place yet
                reference = places[reference_index]

                # Identify reference anchor (start/end/center)
                if place.from_anchor is not None:
                    reference_anchor = place.from_anchor
                else:
                    reference_anchor = refer

                # Absolute s coordinate of the reference anchor point
                s_reference = s_start_by_index[reference_index] + _anchor_offset(
                    reference_anchor,
                    length_by_name[reference.name],
                )
                # Component anchor (start/end/center)
                anchor = refer if place.anchor is None else place.anchor
                # Absolute location of the component
                s_start = s_reference + at - _anchor_offset(anchor, self_length)
                from_name = str(table.name[reference_index])
                from_anchor = place.from_anchor

            s_start_by_index[index] = s_start
            from_name_by_index[index] = from_name
            from_anchor_by_index[index] = from_anchor
            resolved_index_by_name[place.name] = index
            made_progress = True

    unresolved_indices = [
        index for index, s_start in enumerate(s_start_by_index) if s_start is None
    ]
    if unresolved_indices:
        if not diagnostics:
            raise ValueError(
                'Could not resolve all placement positions. Call '
                'Composer.validate() or enable diagnostics for details.'
            )
        _raise_resolution_error_with_diagnostics(places, unresolved_indices)

    table['s_start'] = np.array(s_start_by_index)
    table['s_center'] = table['s_start'] + table['length'] / 2
    table['s_end'] = table['s_start'] + table['length']
    table['s'] = table['s_start'].copy()
    table['from_'] = np.array(from_name_by_index)
    table['from_anchor'] = np.array(from_anchor_by_index)
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


def _raise_resolution_error_with_diagnostics(places, unresolved_indices):
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
