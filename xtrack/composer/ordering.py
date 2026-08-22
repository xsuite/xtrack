"""Stable longitudinal and dependency ordering for resolved placements."""

from dataclasses import dataclass
from functools import cmp_to_key
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ResolvedPlacement:
    """Absolute placement used by coordinate ordering."""

    source_index: int
    name_with_repetition: str
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
            name_with_repetition=source.name[index],
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
    placements_by_center = [placements[index] for index in center_order]

    # Only thin elements can belong to the same-s group.
    same_s_groups = _group_elements_at_same_s(placements_by_center, s_tol)
    group_index_by_name = {
        placement.name_with_repetition: group_index
        for group_index, group in enumerate(same_s_groups)
        for placement in group
    }

    # Order the thin elements at each s according to their placement dependencies.
    ordered_placements = []
    for group_index, group in enumerate(same_s_groups):
        ordered_placements.extend(
            _order_elements_at_same_s(
                group,
                group_index,
                group_index_by_name,
                allow_non_existent_from,
            )
        )

    place_order = [placement.source_index for placement in ordered_placements]
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


def _group_elements_at_same_s(placements, s_tol):
    """Group thin elements that share a longitudinal center coordinate."""
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


def _order_elements_at_same_s(
    group,
    group_index,
    group_index_by_name,
    allow_non_existent_from,
):
    """Order the thin elements in one same-s group. Criteria:

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
    """

    if len(group) == 1 or all(placement.from_ is None for placement in group):
        return group

    from_upstream = []  # placement._from refers to an element before the group
    from_downstream = []  # placement._from refers to an element after the group
    from_same_s = []  # placement._from refers to an element inside group
    unconstrained = []  # no constraint from placement._from

    for placement in group:
        from_name = placement.from_
        if from_name is None:
            unconstrained.append(placement)
            continue
        if from_name not in group_index_by_name:
            if allow_non_existent_from:  # used in Line.insert
                unconstrained.append(placement)
                continue
            raise ValueError(
                f'Component {placement.source_index} '
                f'({placement.name_with_repetition!r}) references missing element '
                f'{from_name!r}.'
            )

        from_group_index = group_index_by_name[from_name]
        if from_group_index < group_index:
            from_upstream.append(placement)
        elif from_group_index > group_index:
            from_downstream.append(placement)
        elif placement.from_anchor is None:
            unconstrained.append(placement)
        else:
            from_same_s.append(placement)

    # References to upstream elements go first, unconstrained elements retain
    # their input order, and references to downstream elements go last.
    # (elements with from_ inside the group are handled below.)
    order = from_upstream + unconstrained + from_downstream

    # Turn same-s placement constraints into insertion instructions to be
    # executed later. For example, Place('b', from_='a', from_anchor='start')
    # stores the placement of b under elements_to_insert_before['a'].
    elements_to_insert_before = {}
    elements_to_insert_after = {}
    for placement in from_same_s:
        target_name = placement.from_
        if placement.from_anchor in ('start', 'center', 'centre'):
            if target_name not in elements_to_insert_before:
                elements_to_insert_before[target_name] = []
            elements_to_insert_before[target_name].append(placement)
        elif placement.from_anchor == 'end':
            if target_name not in elements_to_insert_after:
                elements_to_insert_after[target_name] = []
            elements_to_insert_after[target_name].append(placement)
        else:
            raise ValueError(f'Unknown from_anchor {placement.from_anchor}')

    # Walk the current order and insert the collected elements around their
    # targets. Repeat because an inserted element can itself be another target.
    while elements_to_insert_before or elements_to_insert_after:
        new_order = []
        for placement in order:
            name_with_repetition = placement.name_with_repetition
            new_order.extend(elements_to_insert_before.pop(name_with_repetition, []))
            new_order.append(placement)
            new_order.extend(elements_to_insert_after.pop(name_with_repetition, []))

        if len(new_order) == len(order):
            raise ValueError(
                'Could not sort elements within group; possible circular '
                'dependency in from_ specifications'
            )
        order = new_order

    return order


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
