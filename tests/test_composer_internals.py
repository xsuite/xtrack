import numpy as np
import pytest

import xtrack as xt
from xtrack.composer.components import _all_places
from xtrack.composer.ordering import _sort_places
from xtrack.composer.positions import (
    _anchor_offset,
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


def test_flatten_does_not_mutate_named_line_place():
    env = xt.Environment()
    env.new('marker', xt.Marker)
    env.new_line(name='subline', components=['marker'])
    place = xt.Place('subline', at=2, anchor='start')

    flattened = xt.Composer(env, components=[place]).flatten()

    assert place.name == 'subline'
    assert flattened.components[0].name == 'marker'


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
    assert np.array_equal(table.env_name, ['b', 'a'])
    assert np.allclose(table.s_start, [7, 5])


def test_position_resolution_rejects_from_without_at():
    env = xt.Environment()
    env.new('marker', xt.Marker)
    place = xt.Place('marker', from_='marker')

    with pytest.raises(ValueError, match='without providing `at`'):
        _resolve_s_positions([place], env)


def test_position_resolution_rejects_from_anchor_without_from():
    env = xt.Environment()
    env.new('marker', xt.Marker)
    place = xt.Place('marker', at=1, from_anchor='end')

    with pytest.raises(ValueError, match='without providing `from_`'):
        _resolve_s_positions([place], env)


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


def test_ordering_preserves_input_order_when_from_anchor_is_omitted():
    table = _placement_table(
        name=['dependent', 'base'],
        env_name=['dependent', 'base'],
        s_start=[0.0, 0.0],
        s_center=[0.0, 0.0],
        s_end=[0.0, 0.0],
        length=[0.0, 0.0],
        isthick=[False, False],
        from_=['base', None],
        from_anchor=[None, None],
    )

    sorted_table = _sort_places(table)

    assert np.array_equal(sorted_table.name, ['dependent', 'base'])


def test_omitted_from_anchor_remains_unconstrained_in_a_sandwich():
    table = _placement_table(
        name=['unconstrained', 'base', 'after'],
        env_name=['unconstrained', 'base', 'after'],
        s_start=[0.0, 0.0, 0.0],
        s_center=[0.0, 0.0, 0.0],
        s_end=[0.0, 0.0, 0.0],
        length=[0.0, 0.0, 0.0],
        isthick=[False, False, False],
        from_=['base', None, 'base'],
        from_anchor=[None, None, 'end'],
    )

    sorted_table = _sort_places(table)

    assert np.array_equal(sorted_table.name, ['unconstrained', 'base', 'after'])


def test_external_references_order_coincident_elements_without_anchor():
    table = _placement_table(
        name=['upstream', 'base', 'from_downstream', 'from_upstream', 'downstream'],
        env_name=[
            'upstream',
            'base',
            'from_downstream',
            'from_upstream',
            'downstream',
        ],
        s_start=[0.0, 1.0, 1.0, 1.0, 2.0],
        s_center=[0.0, 1.0, 1.0, 1.0, 2.0],
        s_end=[0.0, 1.0, 1.0, 1.0, 2.0],
        length=[0.0] * 5,
        isthick=[False] * 5,
        from_=[None, None, 'downstream', 'upstream', None],
        from_anchor=[None] * 5,
    )

    sorted_table = _sort_places(table)

    assert np.array_equal(
        sorted_table.name,
        ['upstream', 'from_upstream', 'base', 'from_downstream', 'downstream'],
    )


def test_ordering_sequential_repeated_elements_uses_occurrence_name():
    env = xt.Environment()
    env.new('marker', xt.Marker)
    composer = xt.Composer(
        env,
        components=[xt.Place('marker'), xt.Place('marker')],
    )

    table = composer.resolve_s_positions()

    assert np.array_equal(table.name, ['marker::0', 'marker::1'])
    assert np.array_equal(table.from_, [None, 'marker::0'])


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


def test_ordering_does_not_group_thin_element_after_thick_element():
    table = _placement_table(
        name=['thick', 'thin'],
        env_name=['thick', 'thin'],
        s_start=[-1.0, 0.0],
        s_center=[0.0, 0.0],
        s_end=[1.0, 0.0],
        length=[2.0, 0.0],
        isthick=[True, False],
        from_=[None, 'thick'],
        from_anchor=[None, 'start'],
    )

    sorted_table = _sort_places(table)

    assert np.array_equal(sorted_table.name, ['thick', 'thin'])


def test_composer_build_covers_sequential_and_positioned_paths():
    env = xt.Environment()
    env.new('drift', xt.Drift, length=1)

    sequential = xt.Composer(env, components=['drift'], length=3).build()
    positioned = xt.Composer(
        env,
        components=[xt.Place('drift', at=0, anchor='start')],
        length=3,
        s_tol=1e-12,
    ).build()

    assert sequential.element_names[0] == 'drift'
    assert positioned.element_names[0] == 'drift'
    assert len(sequential.element_names) == len(positioned.element_names) == 2


def test_composer_build_rejects_name():
    env = xt.Environment()
    composer = xt.Composer(env)

    with pytest.raises(ValueError, match='name.*no longer supported'):
        composer.build(name='line')


def test_composer_build_overlap_error_identifies_component():
    env = xt.Environment()
    env.new('a', xt.Drift, length=2)
    env.new('b', xt.Drift, length=2)

    with pytest.raises(ValueError) as error:
        xt.Composer(
            env,
            components=[
                xt.Place('a', at=0, anchor='start'),
                xt.Place('b', at=1, anchor='start'),
            ],
            s_tol=1e-12,
        ).build()

    assert "Overlap before component 1 ('b')" in str(error.value)
