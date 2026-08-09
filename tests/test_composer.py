import copy

import pytest

import xtrack as xt


def test_composer_numeric_length_roundtrip():
    env = xt.Environment()
    env.new('marker', xt.Marker)

    composer = xt.Composer(env, components=['marker'], length=7.0)
    restored = xt.Composer.from_dict(composer.to_dict(), env)

    assert restored.length == 7.0
    assert restored.build().get_length() == 7.0


@pytest.mark.parametrize('as_ref', [False, True])
def test_composer_expression_length_roundtrip(as_ref):
    env = xt.Environment()
    env['line_length'] = 7.0
    length = env.ref['line_length'] if as_ref else 'line_length'
    composer = xt.Composer(env, length=length)

    restored = xt.Composer.from_dict(composer.to_dict(), env)

    assert restored.build().get_length() == 7.0


@pytest.mark.parametrize('length_key', ['l', 'length'])
def test_composer_accepts_length_serialization_aliases(length_key):
    env = xt.Environment()
    data = {
        '__class__': 'Composer',
        'components': [],
        length_key: 3.0,
    }

    restored = xt.Composer.from_dict(data, env)

    assert restored.length == 3.0


def test_empty_composer_can_resolve_positions():
    env = xt.Environment()
    composer = xt.Composer(env)

    table = composer.resolve_s_positions()

    assert len(table) == 0


def test_resolving_line_components_does_not_mutate_place():
    env = xt.Environment()
    env.new('marker', xt.Marker)
    env.new_line(name='subline', components=['marker'])
    place = xt.Place('subline', at=0, anchor='start')
    composer = xt.Composer(env, components=[place])

    composer.resolve_s_positions()

    assert place.name == 'subline'


def test_composer_from_dict_does_not_mutate_input():
    env = xt.Environment()
    env.new('marker', xt.Marker)
    subline = env.new_line(components=['marker'])
    data = xt.Composer(
        env,
        components=[xt.Place(subline, at=0, anchor='start')],
    ).to_dict()
    data_before = copy.deepcopy(data)

    xt.Composer.from_dict(data, env)

    assert data == data_before


def test_cyclic_position_dependencies_are_reported():
    env = xt.Environment()
    env.new('a', xt.Marker)
    env.new('b', xt.Marker)
    composer = xt.Composer(
        env,
        components=[
            xt.Place('a', at=0, from_='b'),
            xt.Place('b', at=0, from_='a'),
        ],
    )

    with pytest.raises(ValueError, match='Could not resolve all s positions'):
        composer.resolve_s_positions()


@pytest.mark.parametrize('field', ['anchor', 'from_anchor'])
def test_place_rejects_unknown_anchor(field):
    with pytest.raises(ValueError, match='anchor'):
        xt.Place('marker', **{field: 'unknown'})
