from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import xtrack as xt
from xtrack.composer.components import _all_places, _expand_components
from xtrack.composer.models import PlacementSpec, ResolvedPlacement
from xtrack.composer.ordering import _sort_places
from xtrack.composer.pipeline import _build_element_names
from xtrack.composer.positions import (
    _anchor_offset,
    _placement_specs_from_places,
    _resolve_placement_records,
    _resolve_s_positions,
)


def _placement_table(**columns):
    return xt.Table({name: np.array(values) for name, values in columns.items()})


def test_all_places_consumes_nested_generator_once():
    inner = (component for component in ['a', xt.Place('b', at=2)])
    components = (component for component in [inner])

    places = _all_places([components])

    assert [place.name for place in places] == ['a', 'b']
    assert places[1].at == 2


def test_expand_components_does_not_mutate_named_line_place():
    env = xt.Environment()
    env.new('marker', xt.Marker)
    env.new_line(name='subline', components=['marker'])
    place = xt.Place('subline', at=2, anchor='start')

    expanded = _expand_components(env, [place])

    assert place.name == 'subline'
    assert expanded[0].name == 'marker'


def test_placement_specs_are_an_immutable_copy_boundary():
    place = xt.Place('marker', at=2, anchor='end')

    specs = _placement_specs_from_places([place])

    assert specs == [
        PlacementSpec(
            source_index=0,
            name='marker',
            at=2,
            from_=None,
            anchor='end',
            from_anchor=None,
        )
    ]
    with pytest.raises(FrozenInstanceError):
        specs[0].name = 'changed'
    assert place.name == 'marker'


@pytest.mark.parametrize(
    ('anchor', 'expected'),
    [('start', 0), ('center', 2), ('centre', 2), ('end', 4)],
)
def test_anchor_offset(anchor, expected):
    assert _anchor_offset(anchor, 4) == expected


def test_position_resolution_supports_forward_dependencies():
    env = xt.Environment()
    env.new('a', xt.Marker)
    env.new('b', xt.Marker)
    places = [
        xt.Place('b', at=2, from_='a'),
        xt.Place('a', at=5),
    ]

    table = _resolve_s_positions(places, env)
    _, records = _resolve_placement_records(places, env, refer='center')

    assert np.array_equal(table.env_name, ['b', 'a'])
    assert np.allclose(table.s_start, [7, 5])
    assert all(isinstance(item, ResolvedPlacement) for item in records)


def test_missing_reference_identifies_root_and_blocked_components():
    env = xt.Environment()
    env.new('a', xt.Marker)
    env.new('b', xt.Marker)
    places = [xt.Place('a', at=0, from_='missing'), xt.Place('b')]

    with pytest.raises(ValueError) as error:
        _resolve_s_positions(places, env, diagnostics=True)

    message = str(error.value)
    assert 'Missing placement reference' in message
    assert "component 0 ('a') references missing element 'missing'" in message
    assert "component 1 ('b')" in message


def test_position_resolution_does_not_annotate_public_places():
    env = xt.Environment()
    env.new('drift', xt.Drift, length=1)
    place = xt.Place('drift')

    table = _resolve_s_positions([place, place], env)

    assert np.array_equal(table.env_name, ['drift', 'drift'])
    assert np.allclose(table.s_start, [0, 1])
    assert place.at is None
    assert place.from_ is None
    assert place.from_anchor is None


def test_ordering_places_start_and_end_dependencies_around_reference():
    table = _placement_table(
        name=['base', 'after', 'before'],
        env_name=['base', 'after', 'before'],
        s_start=[0.0, 0.0, 0.0],
        s_center=[0.0, 0.0, 0.0],
        s_end=[0.0, 0.0, 0.0],
        length=[0.0, 0.0, 0.0],
        isthick=[False, False, False],
        from_=[None, 'base', 'base'],
        from_anchor=[None, 'end', 'start'],
    )

    sorted_table = _sort_places(table)

    assert np.array_equal(sorted_table.name, ['before', 'base', 'after'])
    assert np.array_equal(table.name, ['base', 'after', 'before'])
    assert 'i_place' not in table._col_names


def test_ordering_can_tolerate_removed_reference():
    table = _placement_table(
        name=['base', 'marker'],
        env_name=['base', 'marker'],
        s_start=[0.0, 0.0],
        s_center=[0.0, 0.0],
        s_end=[0.0, 0.0],
        length=[0.0, 0.0],
        isthick=[False, False],
        from_=[None, 'removed'],
        from_anchor=[None, 'start'],
    )

    with pytest.raises(ValueError, match='removed'):
        _sort_places(table)
    assert len(_sort_places(table, allow_non_existent_from=True)) == 2


@pytest.mark.parametrize(
    ('offset', 'expected'),
    [(0.5e-10, ['before', 'base']), (2e-10, ['base', 'before'])],
)
def test_ordering_tolerance_boundary(offset, expected):
    table = _placement_table(
        name=['base', 'before'],
        env_name=['base', 'before'],
        s_start=[0.0, offset],
        s_center=[0.0, offset],
        s_end=[0.0, offset],
        length=[0.0, 0.0],
        isthick=[False, False],
        from_=[None, 'base'],
        from_anchor=[None, 'start'],
    )

    sorted_table = _sort_places(table, s_tol=1e-10)

    assert np.array_equal(sorted_table.name, expected)


def test_pipeline_covers_sequential_and_positioned_paths():
    env = xt.Environment()
    env.new('drift', xt.Drift, length=1)

    sequential = _build_element_names(
        env, ['drift'], refer='center', length=3, s_tol=1e-12
    )
    positioned = _build_element_names(
        env,
        [xt.Place('drift', at=0, anchor='start')],
        refer='center',
        length=3,
        s_tol=1e-12,
    )

    assert sequential[0] == 'drift'
    assert positioned[0] == 'drift'
    assert len(sequential) == len(positioned) == 2


def test_pipeline_overlap_error_identifies_component():
    env = xt.Environment()
    env.new('a', xt.Drift, length=2)
    env.new('b', xt.Drift, length=2)

    with pytest.raises(ValueError) as error:
        _build_element_names(
            env,
            [
                xt.Place('a', at=0, anchor='start'),
                xt.Place('b', at=1, anchor='start'),
            ],
            refer='center',
            length=None,
            s_tol=1e-12,
        )

    assert "Overlap before component 1 ('b')" in str(error.value)
