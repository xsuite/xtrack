"""Component expansion, placement resolution, ordering, and drift creation."""

from collections.abc import Iterable
from functools import cmp_to_key

import numpy as np
import xdeps as xd
import xtrack as xt


_ALLOWED_ANCHORS = (None, 'center', 'centre', 'start', 'end')


def _evaluate_length(env, length):
    """Evaluate a configured line length without changing the specification."""
    if isinstance(length, str):
        return env.eval(length)
    if xd.refs.is_ref(length):
        return length._value
    return length


def _expand_components(env, components, refer='center'):
    """Resolve named lines and recursively expand all nested components."""
    components = _resolve_lines_in_components(components, env)
    return _flatten_components(env, components, refer=refer)


def _build_element_names(env, components, refer, length, s_tol):
    """Run placement and drift generation for already-expanded components."""
    if all(isinstance(component, str) for component in components):
        return _build_sequential_element_names(
            env, components, length=length, s_tol=s_tol
        )

    places = _all_places(components)
    positions = _resolve_s_positions(places, env, refer=refer)
    positions = _sort_places(positions)
    return _generate_element_names_with_drifts(
        env, positions, length=length, s_tol=s_tol
    )


def _build_sequential_element_names(env, components, length, s_tol):
    """Build the optimized path used when no explicit placements are present."""
    element_names = list(map(str, components))
    if length is None:
        return element_names

    components_length = env.new_line(components=element_names).get_length()
    if components_length > length + s_tol:
        raise ValueError(
            f'Line length {components_length} is greater than the requested '
            f'length {length}'
        )
    if components_length < length - s_tol:
        drift = env.new(
            env._get_a_drift_name(),
            xt.Drift,
            length=length - components_length,
        )
        element_names.append(drift)
    return element_names


def _flatten_components(env, components, refer='center'):
    if refer not in ['start', 'center', 'centre', 'end']:
        raise ValueError(
            f'Allowed values for refer are "start", "center" and "end". Got "{refer}".'
        )

    flattened = []
    for component in components:
        this_line = None
        anchor = None
        if isinstance(component, xt.Place) and isinstance(component.name, xt.Line):
            this_line = component.name
            anchor = component.anchor
        if isinstance(component, str) and isinstance(env[component], xt.Line):
            this_line = env[component]
        if isinstance(component, xt.Place) and component.name in env.lines:
            this_line = env.lines[component.name]
            anchor = component.anchor

        if this_line is not None:
            if isinstance(this_line, xt.Composer):
                this_line = this_line.build(name=None, inplace=False)
            elif isinstance(this_line, xt.Line) and this_line.mode == 'compose':
                this_line = this_line.composer.build(name=None, inplace=False)

            if anchor is None:
                anchor = refer or 'center'
            if not this_line.element_names:
                continue

            sub_components = list(this_line.element_names)
            if component.at is not None:
                if isinstance(component.at, str):
                    at = this_line._xdeps_eval.eval(component.at)
                else:
                    at = component.at
                at_of_start = at - _anchor_offset(anchor, this_line.get_length())
                sub_components[0] = xt.Place(
                    sub_components[0],
                    at=at_of_start,
                    anchor='start',
                    from_=component.from_,
                    from_anchor=component.from_anchor,
                )
            flattened.extend(sub_components)
        elif isinstance(component, xt.Composer):
            flattened.extend(component.build(inplace=False).element_names)
        elif isinstance(component, xt.Line):
            if component.mode == 'compose':
                component = component.composer.build(name=None, inplace=False)
            flattened.extend(component.element_names)
        elif isinstance(component, Iterable) and not isinstance(component, str):
            flattened.extend(_flatten_components(env, component, refer=refer))
        else:
            flattened.append(component)

    return flattened


def _all_places(sequence):
    places = []
    for component in sequence:
        if isinstance(component, xt.Place):
            places.append(component)
        elif not isinstance(component, str) and hasattr(component, '__iter__'):
            for nested_component in component:
                if isinstance(nested_component, xt.Place):
                    break
                if not isinstance(nested_component, (str, xt.Line)):
                    raise TypeError(
                        'Only places, elements, strings or Lines are allowed '
                        'in sequences'
                    )
            places.extend(_all_places(component))
        else:
            if not isinstance(component, (str, xt.Line)):
                raise TypeError(
                    'Only places, elements, strings or Lines are allowed in sequences'
                )
            places.append(xt.Place(component, at=None, from_=None))
    return places


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
    index, places, start_by_place, place_by_name, lengths, evaluator, refer
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
                index, places, start_by_place, place_by_name, lengths, evaluator, refer
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
    places = [place.copy() for place in seq_all_places]
    aux_line, table, lengths = _prepare_position_table(places, env, refer)
    if not places:
        return _add_empty_position_columns(table)

    start_by_place = _resolve_place_coordinates(
        places, lengths, aux_line._xdeps_eval, refer
    )
    return _add_resolved_position_columns(table, places, start_by_place)


def _assign_position_groups(table, s_tol):
    """Label adjacent elements that share a longitudinal center coordinate."""
    group_ids = np.zeros(len(table), dtype=int)
    for index in range(1, len(table)):
        different_center = (
            abs(table.s_center[index] - table.s_center[index - 1]) > s_tol
        )
        overlapping_thick_element = (
            table.isthick[index] and table.s_end[index] - table.s_start[index] != 0
        )
        group_ids[index] = group_ids[index - 1]
        if different_center or overlapping_thick_element:
            group_ids[index] += 1
    table['group_id'] = group_ids
    return table


def _iter_group_bounds(group_ids):
    """Yield half-open index ranges for consecutive equal group identifiers."""
    start = 0
    while start < len(group_ids):
        end = start + 1
        while end < len(group_ids) and group_ids[end] == group_ids[start]:
            end += 1
        yield start, end
        start = end


def _classify_group_dependencies(
    group, name_index, group_start, group_end, allow_non_existent_from
):
    """Partition a coincident group according to dependency location."""
    from_before = []
    from_after = []
    from_inside = []
    no_from = []

    for index, from_name in enumerate(group.from_):
        if from_name is None:
            no_from.append(index)
            continue
        if from_name not in name_index:
            if allow_non_existent_from:
                no_from.append(index)
                continue
            raise ValueError(f'Element {from_name} not found in the line')

        from_index = name_index[from_name]
        if from_index < group_start:
            from_before.append(index)
        elif from_index >= group_end:
            from_after.append(index)
        else:
            from_inside.append(index)

    base_order = from_before + no_from + from_after
    return base_order, from_inside


def _build_group_insertions(group, from_inside):
    """Collect within-group insertions keyed by their reference element."""
    insert_before = {}
    insert_after = {}
    for index in from_inside:
        from_name = group.from_[index]
        from_anchor = group.from_anchor[index]

        # Within a thin sandwich, center and an omitted anchor behave as start.
        if from_anchor in (None, 'start', 'center', 'centre'):
            insert_before.setdefault(from_name, []).append(index)
        elif from_anchor == 'end':
            insert_after.setdefault(from_name, []).append(index)
        else:
            raise ValueError(f'Unknown from_anchor {from_anchor}')
    return insert_before, insert_after


def _apply_group_insertions(group, base_order, insert_before, insert_after):
    """Apply dependent insertions, detecting circular specifications."""
    order = base_order.copy()
    while insert_before or insert_after:
        new_order = []
        for index in order:
            name = group.name[index]
            new_order.extend(insert_before.pop(name, []))
            new_order.append(index)
            new_order.extend(insert_after.pop(name, []))

        if len(new_order) == len(order):
            raise ValueError(
                'Could not sort elements within group; possible circular '
                'dependency in from_ specifications'
            )
        order = new_order
    return order


def _order_coincident_group(table, start, end, name_index, allow_non_existent_from):
    """Return original place indices in the required order for one group."""
    if end - start == 1:
        return [table.i_place[start]]
    if all(anchor is None for anchor in table.from_anchor[start:end]):
        return list(table.i_place[start:end])

    group = table.rows[start:end]
    base_order, from_inside = _classify_group_dependencies(
        group, name_index, start, end, allow_non_existent_from
    )
    insert_before, insert_after = _build_group_insertions(group, from_inside)
    order = _apply_group_insertions(group, base_order, insert_before, insert_after)
    return list(group.rows[order].i_place)


def _add_upstream_gaps(table):
    """Recompute coordinate columns and the gap before each sorted element."""
    table['s_center'] = table['s_start'] + table['length'] / 2
    table['s_end'] = table['s_start'] + table['length']
    table['ds_upstream'] = 0 * table['s_start']
    table['ds_upstream'][1:] = table['s_start'][1:] - table['s_end'][:-1]
    table['ds_upstream'][0] = table['s_start'][0]
    table['s'] = table['s_start']
    return table


def _sort_places(tt_unsorted, s_tol=1e-10, allow_non_existent_from=False):
    """Sort placements by position and dependency order without mutating input."""
    source = tt_unsorted.rows[:]
    source['i_place'] = np.arange(len(source))

    if not len(source):
        source['group_id'] = np.array([], dtype=int)
        source['ds_upstream'] = np.array([], dtype=float)
        return source

    center_order = _argsort_s(source.s_center, tol=s_tol)
    center_sorted = _assign_position_groups(source.rows[center_order], s_tol)

    # Caching these indices avoids repeated Table row lookups in large lines.
    name_index = {name: index for index, name in enumerate(center_sorted.name)}
    place_order = []
    for start, end in _iter_group_bounds(center_sorted.group_id):
        place_order.extend(
            _order_coincident_group(
                center_sorted,
                start,
                end,
                name_index,
                allow_non_existent_from,
            )
        )
    return _add_upstream_gaps(source.rows[place_order])


def _generate_element_names_with_drifts(env, tt_sorted, length=None, s_tol=1e-6):
    names_with_drifts = []
    if not len(tt_sorted):
        if length is not None and length > s_tol:
            names_with_drifts.append(env._get_drift(length))
        return list(map(str, names_with_drifts))

    for index, name in enumerate(tt_sorted.env_name):
        gap = tt_sorted['ds_upstream', index]
        if np.abs(gap) > s_tol:
            if gap < 0:
                raise ValueError(f'Negative drift length: {gap}, upstream of {name}')
            names_with_drifts.append(env._get_drift(gap))
        names_with_drifts.append(name)

    if length is not None:
        line_length = tt_sorted['s_end'][-1]
        if line_length > length + s_tol:
            raise ValueError(
                f'Line length {line_length} is greater than the requested '
                f'length {length}'
            )
        if line_length < length - s_tol:
            names_with_drifts.append(env._get_drift(length - line_length))
    return list(map(str, names_with_drifts))


def _argsort_s(sequence, tol=10e-10):
    """Argsort with a tolerance while retaining Python's stable ordering."""
    sequence_indices = np.arange(len(sequence))

    def comparator(first_index, second_index):
        first = sequence[first_index]
        second = sequence[second_index]
        if np.abs(first - second) < tol:
            return 0
        return -1 if first < second else 1

    return sorted(sequence_indices, key=cmp_to_key(comparator))


def _resolve_lines_in_components(components, env):
    components = list(components)
    for index, component in enumerate(components):
        if (
            isinstance(component, xt.Place)
            and isinstance(component.name, str)
            and component.name in env.lines
        ):
            component = component.copy()
            component.name = env.lines[component.name]
            components[index] = component
        if isinstance(component, str) and component in env.lines:
            components[index] = env.lines[component]
    return components
