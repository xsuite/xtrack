"""Anchor semantics and longitudinal coordinate resolution."""

import numpy as np
import xdeps as xd
import xtrack as xt

from .models import PlacementSpec, ResolvedPlacement


_ALLOWED_ANCHORS = (None, 'center', 'centre', 'start', 'end')


def _resolve_s_positions(seq_all_places, env, refer='center', diagnostics=False):
    """Resolve placement specifications to a table of absolute coordinates."""
    table, placements = _resolve_placement_records(
        seq_all_places,
        env,
        refer=refer,
        diagnostics=diagnostics,
    )
    if not placements:
        table['s_start'] = np.array([], dtype=float)
        table['s_center'] = np.array([], dtype=float)
        table['s_end'] = np.array([], dtype=float)
        table['s'] = np.array([], dtype=float)
        table['from_'] = np.array([], dtype=object)
        table['from_anchor'] = np.array([], dtype=object)
        return table

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


def _placement_specs_from_places(places):
    """Copy public ``Place`` data into immutable pipeline records."""
    return [
        PlacementSpec(
            source_index=index,
            name=place.name,
            at=place.at,
            from_=place.from_,
            anchor=place.anchor,
            from_anchor=place.from_anchor,
        )
        for index, place in enumerate(places)
    ]


def _prepare_position_table(specs, env, refer):
    """Collect element metadata and create the result table skeleton."""
    aux_line = env.new_line(components=[spec.name for spec in specs], refer=refer)
    table = aux_line.get_table()
    table['length'] = np.diff(table.s, append=table.s[-1])
    table = table.rows[:-1]
    lengths = xt.Table({'name': table.env_name, 'length': table.length})
    return aux_line, table, lengths


def _evaluate_position_expression(at, evaluator):
    if isinstance(at, str):
        return evaluator.eval(at)
    if xd.refs.is_ref(at):
        return at._value
    return at


def _resolved_record(spec, table, length, s_start, from_name=None, from_anchor=None):
    """Combine an immutable specification with its resolved coordinates."""
    return ResolvedPlacement(
        source_index=spec.source_index,
        name=spec.name,
        table_name=str(table.name[spec.source_index]),
        env_name=str(table.env_name[spec.source_index]),
        length=length,
        isthick=bool(table.isthick[spec.source_index]),
        s_start=s_start,
        from_=from_name,
        from_anchor=from_anchor,
    )


def _try_resolve_spec(
    index,
    specs,
    resolved_by_index,
    resolved_by_name,
    table,
    lengths,
    evaluator,
    refer,
):
    """Resolve one specification if all of its dependencies are available."""
    spec = specs[index]
    if index in resolved_by_index:
        return False

    if (spec.from_ is not None or spec.from_anchor is not None) and spec.at is None:
        raise ValueError(
            'Cannot specify `from_` or `from_anchor` without providing `at`. '
            f'Error in placement `{spec}`.'
        )

    self_length = lengths['length', spec.name]
    if spec.at is None:
        previous = resolved_by_index.get(index - 1)
        if previous is None:
            return False
        from_name = spec.from_
        from_anchor = spec.from_anchor
        if not str(previous.name).startswith('||drift'):
            from_name = previous.name
            from_anchor = 'end'
        resolved = _resolved_record(
            spec,
            table,
            self_length,
            previous.s_end,
            from_name=from_name,
            from_anchor=from_anchor,
        )
    else:
        at = _evaluate_position_expression(spec.at, evaluator)
        from_length = None
        s_start_from = None
        if spec.from_ is not None:
            reference = resolved_by_name.get(spec.from_)
            if reference is None:
                return False
            from_length = reference.length
            s_start_from = reference.s_start

        s_start = _resolve_one_position(
            at,
            anchor=spec.anchor,
            from_anchor=spec.from_anchor,
            self_length=self_length,
            from_length=from_length,
            s_start_from=s_start_from,
            default_anchor=refer,
        )
        resolved = _resolved_record(
            spec,
            table,
            self_length,
            s_start,
            from_name=spec.from_,
            from_anchor=spec.from_anchor,
        )

    resolved_by_index[index] = resolved
    resolved_by_name[spec.name] = resolved
    return True


def _resolve_spec_coordinates(
    specs, table, lengths, evaluator, refer, diagnostics=False
):
    """Resolve all specifications, iterating until dependencies stop progressing."""
    resolved_by_index = {}
    resolved_by_name = {}

    if specs[0].at is None:
        first = specs[0]
        resolved = _resolved_record(
            first,
            table,
            lengths['length', first.name],
            0,
            from_name=first.from_,
            from_anchor=first.from_anchor,
        )
        resolved_by_index[0] = resolved
        resolved_by_name[first.name] = resolved

    made_progress = True
    while made_progress:
        made_progress = False
        for index in range(len(specs)):
            if _try_resolve_spec(
                index,
                specs,
                resolved_by_index,
                resolved_by_name,
                table,
                lengths,
                evaluator,
                refer,
            ):
                made_progress = True

    if len(resolved_by_index) != len(specs):
        if not diagnostics:
            raise ValueError(
                'Could not resolve all placement positions. Call '
                'Composer.validate() or enable diagnostics for details.'
            )
        unresolved = [
            spec for spec in specs if spec.source_index not in resolved_by_index
        ]
        _raise_resolution_error(specs, unresolved)
    return [resolved_by_index[index] for index in range(len(specs))]


def _format_spec(spec):
    return f'component {spec.source_index} ({spec.name!r})'


def _find_dependency_cycle(specs, unresolved):
    """Return one cycle of component indices, including its repeated endpoint."""
    unresolved_indices = {spec.source_index for spec in unresolved}
    indices_by_name = {}
    for spec in specs:
        indices_by_name.setdefault(spec.name, []).append(spec.source_index)

    dependency_by_index = {}
    for spec in unresolved:
        dependency = None
        if spec.from_ is not None:
            candidates = indices_by_name.get(spec.from_, [])
            dependency = next(
                (index for index in candidates if index in unresolved_indices),
                None,
            )
        elif spec.at is None and spec.source_index > 0:
            dependency = spec.source_index - 1
        if dependency in unresolved_indices:
            dependency_by_index[spec.source_index] = dependency

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


def _raise_resolution_error(specs, unresolved):
    """Raise a specific diagnostic for a stalled dependency resolution."""
    available_names = {spec.name for spec in specs}
    missing = [
        spec
        for spec in unresolved
        if spec.from_ is not None and spec.from_ not in available_names
    ]
    if missing:
        details = '; '.join(
            f'{_format_spec(spec)} references missing element {spec.from_!r}'
            for spec in missing
        )
        blocked = ', '.join(_format_spec(spec) for spec in unresolved)
        raise ValueError(f'Missing placement reference: {details}. Blocked: {blocked}.')

    cycle = _find_dependency_cycle(specs, unresolved)
    if cycle is not None:
        specs_by_index = {spec.source_index: spec for spec in specs}
        chain = ' -> '.join(_format_spec(specs_by_index[index]) for index in cycle)
        raise ValueError(f'Cyclic placement dependency: {chain}.')

    blocked = ', '.join(_format_spec(spec) for spec in unresolved)
    raise ValueError(f'Could not resolve placement dependencies: {blocked}.')


def _resolve_placement_records(places, env, refer='center', diagnostics=False):
    """Resolve public places and return the table skeleton plus immutable records."""
    specs = _placement_specs_from_places(places)
    aux_line, table, lengths = _prepare_position_table(specs, env, refer)
    if not specs:
        return table, []
    placements = _resolve_spec_coordinates(
        specs,
        table,
        lengths,
        aux_line._xdeps_eval,
        refer,
        diagnostics=diagnostics,
    )
    return table, placements
