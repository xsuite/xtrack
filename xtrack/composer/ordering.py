"""Stable longitudinal and dependency ordering for resolved placements."""

from functools import cmp_to_key

import numpy as np


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
