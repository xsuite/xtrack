"""Orchestration of component normalization, placement, and materialization."""

from .components import (
    _all_places,
    _build_sequential_element_names,
    _generate_element_names_with_drifts,
)
from .ordering import _sort_places
from .positions import _resolve_s_positions


def _build_element_names(env, components, refer, length, s_tol):
    """Run the complete build pipeline for already-expanded components."""
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
