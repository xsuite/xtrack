"""Stable longitudinal and dependency ordering for resolved placements."""

from dataclasses import dataclass
from functools import cmp_to_key
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ResolvedPlacement:
    """Absolute placement used by coordinate ordering."""

    source_index: int
    name: Any
    table_name: str
    env_name: str
    length: Any
    isthick: bool
    s_start: Any
    from_: str | None
    from_anchor: str | None

    @property
    def s_center(self):
        return self.s_start + self.length / 2

    @property
    def s_end(self):
        return self.s_start + self.length


def _sort_places(tt_unsorted, s_tol=1e-10, allow_non_existent_from=False):
    """Sort a placement table without mutating it.

    Notes
    -----
    The following ordering rules are applied:

    - Components are sorted by increasing ``s_center``. Only thin elements can
      form a group at the same ``s``; for these elements ``s_start``, ``s_center``,
      and ``s_end`` are identical. This is why ``s_center`` is used as the common
      sorting coordinate.
    - For thin elements at the same ``s`` (within ``s_tol``):
      - input order is preserved unless placement dependencies establish an order;
      - an element whose ``from_`` names an upstream element moves toward the
        beginning of the group;
      - an element whose ``from_`` names a downstream element moves toward the end
        of the group;
      - when ``from_`` names an element inside the group:
        - an explicit ``from_anchor`` of ``'start'``, ``'center'``, or ``'centre'``
          places the element before the element named by ``from_``;
        - an explicit ``from_anchor`` of ``'end'`` places it after the element named
          by ``from_``;
        - an omitted ``from_anchor`` imposes no tie-break;
      - sequential elements depend on the end of the previous occurrence.
    """
    source = tt_unsorted.rows[:]
    if not len(source):
        source['i_place'] = np.array([], dtype=int)
        source['group_id'] = np.array([], dtype=int)
        source['ds_upstream'] = np.array([], dtype=float)
        return source

    placements = [
        ResolvedPlacement(
            source_index=index,
            name=source.env_name[index],
            table_name=source.name[index],
            env_name=source.env_name[index],
            length=source.length[index],
            isthick=bool(source.isthick[index]),
            s_start=source.s_start[index],
            from_=source.from_[index],
            from_anchor=source.from_anchor[index],
        )
        for index in range(len(source))
    ]
    # Sort placements by longitudinal center.
    center_order = _argsort_s(
        [placement.s_center for placement in placements],
        tol=s_tol,
    )
    center_sorted = [placements[index] for index in center_order]

    # Group placements sharing the same longitudinal center (each group is a
    # list of ResolvedPlacement objects). Only groups made only of thin elements
    # can contain more than one placement.
    groups = _group_by_position(center_sorted, s_tol)

    name_index = {
        placement.table_name: index for index, placement in enumerate(center_sorted)
    }

    # Order coincident placements according to their dependencies.
    placements = []
    group_start = 0
    for group in groups:
        placements.extend(
            _order_coincident_group(
                group,
                group_start,
                name_index,
                allow_non_existent_from,
            )
        )
        group_start += len(group)

    place_order = [placement.source_index for placement in placements]
    sorted_table = source.rows[place_order]
    sorted_table['i_place'] = np.array(place_order)
    sorted_table['s_center'] = sorted_table['s_start'] + sorted_table['length'] / 2
    sorted_table['s_end'] = sorted_table['s_start'] + sorted_table['length']
    sorted_table['ds_upstream'] = 0 * sorted_table['s_start']
    sorted_table['ds_upstream'][1:] = (
        sorted_table['s_start'][1:] - sorted_table['s_end'][:-1]
    )
    sorted_table['ds_upstream'][0] = sorted_table['s_start'][0]
    sorted_table['s'] = sorted_table['s_start']
    return sorted_table


def _group_by_position(placements, s_tol):
    """Group adjacent placements that share a longitudinal center coordinate."""
    if not placements:
        return []

    groups = [[placements[0]]]
    for placement in placements[1:]:
        previous = groups[-1][-1]
        different_center = abs(placement.s_center - previous.s_center) > s_tol
        previous_is_thick = previous.isthick and previous.s_end - previous.s_start != 0
        placement_is_thick = (
            placement.isthick and placement.s_end - placement.s_start != 0
        )
        if different_center or previous_is_thick or placement_is_thick:
            groups.append([placement])
        else:
            groups[-1].append(placement)
    return groups


def _classify_group_dependencies(
    group, name_index, group_start, group_end, allow_non_existent_from
):
    """Partition a coincident group according to dependency location."""
    from_before = []
    from_after = []
    from_inside = []
    no_from = []

    for index, placement in enumerate(group):
        from_name = placement.from_
        if from_name is None:
            no_from.append(index)
            continue
        if from_name not in name_index:
            if allow_non_existent_from:
                no_from.append(index)
                continue
            raise ValueError(
                f'Component {placement.source_index} '
                f'({placement.table_name!r}) references missing element '
                f'{from_name!r}.'
            )

        from_index = name_index[from_name]
        if from_index < group_start:
            from_before.append(index)
        elif from_index >= group_end:
            from_after.append(index)
        elif placement.from_anchor is None:
            no_from.append(index)
        else:
            from_inside.append(index)

    base_order = from_before + no_from + from_after
    return base_order, from_inside


def _build_group_insertions(group, from_inside):
    """Collect within-group insertions keyed by their reference element."""
    insert_before = {}
    insert_after = {}
    for index in from_inside:
        placement = group[index]

        # Within a thin sandwich, center behaves as start.
        if placement.from_anchor in ('start', 'center', 'centre'):
            insert_before.setdefault(placement.from_, []).append(index)
        elif placement.from_anchor == 'end':
            insert_after.setdefault(placement.from_, []).append(index)
        else:
            raise ValueError(f'Unknown from_anchor {placement.from_anchor}')
    return insert_before, insert_after


def _apply_group_insertions(group, base_order, insert_before, insert_after):
    """Apply dependent insertions, detecting circular specifications."""
    order = base_order.copy()
    while insert_before or insert_after:
        new_order = []
        for index in order:
            name = group[index].table_name
            new_order.extend(insert_before.pop(name, []))
            new_order.append(index)
            new_order.extend(insert_after.pop(name, []))

        if len(new_order) == len(order):
            raise ValueError(
                'Could not sort elements within group; possible circular '
                'dependency in from_ specifications'
            )
        order = new_order
    return [group[index] for index in order]


def _order_coincident_group(
    group,
    group_start,
    name_index,
    allow_non_existent_from,
):
    """Apply dependency ordering to one coincident-position group."""
    if len(group) == 1 or all(placement.from_ is None for placement in group):
        return group

    group_end = group_start + len(group)
    base_order, from_inside = _classify_group_dependencies(
        group,
        name_index,
        group_start,
        group_end,
        allow_non_existent_from,
    )
    insert_before, insert_after = _build_group_insertions(group, from_inside)
    return _apply_group_insertions(group, base_order, insert_before, insert_after)


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
