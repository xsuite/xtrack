"""Component expansion, normalization, and drift materialization."""

from collections.abc import Iterable

import numpy as np
import xdeps as xd
import xtrack as xt

from .positions import _anchor_offset


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
    """Recursively replace nested lines and composers with their elements."""
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
    """Normalize nested component sequences to a flat list of places."""
    places = []
    for component in sequence:
        if isinstance(component, xt.Place):
            places.append(component)
        elif isinstance(component, Iterable) and not isinstance(
            component, (str, xt.Line)
        ):
            # Materialize one-shot iterables exactly once at the copy boundary.
            places.extend(_all_places(list(component)))
        else:
            if not isinstance(component, (str, xt.Line)):
                raise TypeError(
                    'Only places, elements, strings or Lines are allowed in sequences'
                )
            places.append(xt.Place(component, at=None, from_=None))
    return places


def _generate_element_names_with_drifts(env, tt_sorted, length=None, s_tol=1e-6):
    """Materialize positive gaps in a sorted placement table as drifts."""
    names_with_drifts = []
    if not len(tt_sorted):
        if length is not None and length > s_tol:
            names_with_drifts.append(env._get_drift(length))
        return list(map(str, names_with_drifts))

    for index, name in enumerate(tt_sorted.env_name):
        gap = tt_sorted['ds_upstream', index]
        if np.abs(gap) > s_tol:
            if gap < 0:
                raise ValueError(
                    f'Overlap before component {index} ({name!r}): previous '
                    f'element extends {-gap} m beyond its start.'
                )
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


def _validate_placement_geometry(
    tt_sorted,
    length=None,
    s_tol=1e-6,
    check_overlaps=True,
):
    """Check resolved overlaps and length constraints without creating drifts."""
    if not len(tt_sorted):
        return

    if check_overlaps:
        for index, name in enumerate(tt_sorted.env_name):
            gap = tt_sorted['ds_upstream', index]
            if gap < -s_tol:
                raise ValueError(
                    f'Overlap before component {index} ({name!r}): previous '
                    f'element extends {-gap} m beyond its start.'
                )

    if length is not None:
        line_length = tt_sorted['s_end'][-1]
        if line_length > length + s_tol:
            raise ValueError(
                f'Line length {line_length} is greater than the requested '
                f'length {length}'
            )


def _resolve_lines_in_components(components, env):
    """Replace named lines without mutating caller-owned placement objects."""
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
