"""Local geometry must not inherit the rounding error of a long prefix."""

from fractions import Fraction
from math import fsum
import json

import numpy as np
import pytest

import xtrack as xt


ANCHORS = {'start': 0, 'center': Fraction(1, 2),
           'centre': Fraction(1, 2), 'end': 1}


def test_long_prefix_reproducer():
    env = xt.Environment()
    env.new('prefix', xt.Drift, length=20000.)
    env.new('q', xt.Quadrupole, length=0.11)
    line = env.new_line(
        components=[xt.Place('prefix', at=0., anchor='start')] + [
            xt.Place('q', at=20001. + i, anchor='start') for i in range(1000)],
        length=21001., s_tol=1e-10)

    table = line.get_table()
    gaps = table.rows.match(name=r'\|\|drift.*').length
    np.testing.assert_array_equal(gaps, [1.] + [0.89] * 1000)
    assert fsum(table.length * table.isthick) == 21001.
    assert env['q'].length == 0.11


@pytest.mark.parametrize('anchor1', ANCHORS)
@pytest.mark.parametrize('anchor2', ANCHORS)
def test_absolute_anchors_are_independent_of_prefix(anchor1, anchor2):
    l1, l2 = Fraction(0.11), Fraction(0.23)
    expected_gap = float(1 + ANCHORS[anchor1] * l1 - ANCHORS[anchor2] * l2 - l1)
    expected_tail = float(1 + ANCHORS[anchor2] * l2 - l2)
    for prefix in [0., 20000., 2.**30]:
        env = xt.Environment()
        env.new('a', xt.Quadrupole, length=float(l1))
        env.new('b', xt.Quadrupole, length=float(l2))
        composer = xt.Composer(env, length=prefix + 3., s_tol=1e-14, components=[
            xt.Place('a', at=prefix + 1., anchor=anchor1),
            xt.Place('b', at=prefix + 2., anchor=anchor2),
        ])
        positions = composer.resolve_s_positions()
        assert positions['ds_upstream', 'b'] == expected_gap
        for column in ['s', 's_start', 's_center', 's_end']:
            assert positions[column].dtype == np.float64
        table = composer.build().get_table()
        gaps = table.rows.match(name=r'\|\|drift.*').length
        assert gaps[1] == expected_gap
        assert gaps[-1] == expected_tail
        assert fsum(table.length * table.isthick) == prefix + 3.


@pytest.mark.parametrize('anchor', ANCHORS)
@pytest.mark.parametrize('from_anchor', ANCHORS)
def test_relative_anchors_and_forward_references(anchor, from_anchor):
    l1, l2, offset = Fraction(0.11), Fraction(0.23), Fraction(0.7)
    expected = float(offset + ANCHORS[from_anchor] * l1 - ANCHORS[anchor] * l2 - l1)
    for prefix in [0., 20000., 2.**30]:
        env = xt.Environment()
        env.new('a', xt.Quadrupole, length=float(l1))
        env.new('b', xt.Quadrupole, length=float(l2))
        env['offset'] = float(offset)
        composer = xt.Composer(env, s_tol=1e-14, components=[
            xt.Place('b', at='offset', from_='a',
                     anchor=anchor, from_anchor=from_anchor),
            xt.Place('a', at=prefix + 1., anchor='start'),
        ])
        positions = composer.resolve_s_positions()
        assert list(positions.env_name) == ['a', 'b']
        assert positions['ds_upstream', 'b'] == expected
        line = composer.build()
        assert line[line.element_names[-2]].length == expected


def test_long_relative_chain_and_sequential_elements():
    env = xt.Environment()
    env.new('q', xt.Quadrupole, length=0.11)
    env.new('marker', xt.Marker)
    composer = xt.Composer(env, s_tol=1e-14, components=[
        xt.Place('q', at=20000., anchor='start'),
        *[xt.Place('q', at=0.3, from_='q', anchor='start', from_anchor='start')
          for _ in range(1000)],
        'marker', 'q', 'marker',
    ])
    positions = composer.resolve_s_positions()
    np.testing.assert_array_equal(positions.ds_upstream[1:1001], [0.19] * 1000)
    np.testing.assert_allclose(positions.ds_upstream[1001:], 0., rtol=0., atol=1e-25)
    assert len(composer.build().element_names) == len(positions) + 1001


@pytest.mark.parametrize('anchor', ANCHORS)
@pytest.mark.parametrize('compose', [False, True])
def test_nested_lines_and_flattened_serialization(anchor, compose):
    env = xt.Environment()
    env.new('a', xt.Quadrupole, length=0.11)
    env.new('b', xt.Quadrupole, length=0.23)
    env.new_line(name='cell', components=['a', 'b'], compose=compose)
    places = [xt.Place('cell', at=20001. + i, anchor=anchor) for i in range(3)]
    specifications = [vars(place).copy() for place in places]
    composer = xt.Composer(env, components=places, s_tol=1e-14, length=20004.)
    flat = composer.flatten()
    data = json.loads(json.dumps(flat.to_dict()))
    for place, serialized in zip(flat.components, data['components']):
        if place.at is not None:
            assert isinstance(place.at, xt.CompensatedFloat)
            assert set(serialized) == {'name', 'at', 'anchor'}
            assert serialized['at'] == place.at.to_dict()
        assert set(vars(place)) == {'name', 'at', 'from_', 'anchor', 'from_anchor', 'env'}
    restored = xt.Composer.from_dict(data, env)
    assert restored.to_dict() == data
    expected_gap = float(1 - Fraction(0.11) - Fraction(0.23))
    expected_tail = float(1 - (1 - ANCHORS[anchor]) * (Fraction(0.11) + Fraction(0.23)))
    for candidate in [composer, flat, restored]:
        table = candidate.build().get_table()
        gaps = table.rows.match(name=r'\|\|drift.*').length
        assert len(gaps) == 4
        np.testing.assert_array_equal(gaps[1:3], [expected_gap] * 2)
        assert gaps[-1] == expected_tail
        assert fsum(table.length * table.isthick) == 20004.
    assert [vars(place) for place in places] == specifications
    assert all(isinstance(place.at, float) for place in places)


def test_nested_line_length_keeps_its_own_rounding_residual():
    env = xt.Environment()
    env.new('q', xt.Quadrupole, length=0.11)
    cell = env.new_line(components=['q'] * 1000)
    line = env.new_line(components=[
        xt.Place(cell, at=20111., anchor='end'),
        xt.Place(cell, at=20222., anchor='end'),
    ], length=20223., s_tol=1e-14)
    gaps = line.get_table().rows.match(name=r'\|\|drift.*').length
    assert len(gaps) == 3
    assert gaps[1] == float(111 - 1000 * Fraction(0.11))
    assert gaps[2] == 1.


@pytest.mark.parametrize('as_ref', [False, True])
def test_nested_line_relative_expression(as_ref):
    env = xt.Environment()
    env.new('origin', xt.Marker)
    env.new('end', xt.Marker)
    env.new('a', xt.Quadrupole, length=0.11)
    env.new('b', xt.Quadrupole, length=0.23)
    cell = env.new_line(components=['a', 'b'])
    env['offset'] = 0.7
    composer = xt.Composer(env, s_tol=1e-14, components=[
        xt.Place('origin', at=20001.),
        xt.Place(cell, at=env.ref['offset'] if as_ref else 'offset',
                 from_='origin', anchor='end'),
        xt.Place('end', at=20002.),
    ])
    for candidate in [composer, composer.flatten()]:
        table = candidate.build().get_table()
        gaps = table.rows.match(name=r'\|\|drift.*').length
        assert len(gaps) == 3
        assert gaps[1] == float(Fraction(0.7) - Fraction(0.11) - Fraction(0.23))
        assert gaps[2] == 1. - 0.7


@pytest.mark.parametrize('placed', [False, True])
def test_sequential_final_drift_uses_compensated_length(placed):
    env = xt.Environment()
    env.new('prefix', xt.Drift, length=20000.)
    env.new('q', xt.Quadrupole, length=0.11)
    components = ['prefix'] + ['q'] * 1000
    if placed:
        components[0] = xt.Place('prefix', at=0., anchor='start')
    line = env.new_line(components=components, length=20111., s_tol=1e-14)
    expected = float(111 - 1000 * Fraction(0.11))
    assert len(line.element_names) == 1002
    assert line[line.element_names[-1]].length == expected
    assert fsum(line.get_table().length) == 20111.


def test_ordering_retains_offsets_below_absolute_coordinate_resolution():
    env = xt.Environment()
    for name in ['a', 'b', 'c']:
        env.new(name, xt.Marker)
    composer = xt.Composer(env, s_tol=1e-12, components=[
        xt.Place('a', at=2.**30),
        xt.Place('b', at=-2e-10, from_='a'),
        xt.Place('c', at=2e-10, from_='a'),
    ])
    table = composer.resolve_s_positions()
    assert list(table.name) == ['b', 'a', 'c']
    np.testing.assert_array_equal(table.ds_upstream[1:], [2e-10, 2e-10])
    line = composer.build()
    assert line[line.element_names[2]].length == 2e-10
    assert line[line.element_names[4]].length == 2e-10


@pytest.mark.parametrize('s_tol, raises', [(1e-12, True), (1e-9, False)])
def test_overlap_tolerance_uses_local_distance(s_tol, raises):
    env = xt.Environment()
    env.new('a', xt.Quadrupole, length=0.11)
    env.new('b', xt.Quadrupole, length=0.23)
    composer = xt.Composer(env, s_tol=s_tol, components=[
        xt.Place('a', at=2.**30, anchor='start'),
        xt.Place('b', at=-2e-10, from_='a@end', anchor='start'),
    ])
    for operation in [composer.validate, composer.build]:
        if raises:
            with pytest.raises(ValueError, match='Overlap'):
                operation()
        else:
            operation()


@pytest.mark.parametrize('s_tol, raises', [(1e-12, True), (1e-9, False)])
def test_final_length_tolerance_uses_local_distance(s_tol, raises):
    env = xt.Environment()
    env.new('q', xt.Quadrupole, length=0.125 + 2e-10)
    composer = xt.Composer(env, s_tol=s_tol, length=2.**30 + 0.125, components=[
        xt.Place('q', at=2.**30, anchor='start'),
    ])
    for operation in [composer.validate, composer.build]:
        if raises:
            with pytest.raises(ValueError, match='greater than the requested'):
                operation()
        else:
            operation()


def test_small_final_gap_is_not_lost_in_absolute_coordinates():
    env = xt.Environment()
    length = 0.125 - 2e-10
    env.new('q', xt.Quadrupole, length=length)
    line = env.new_line(components=[
        xt.Place('q', at=2.**30, anchor='start'),
    ], length=2.**30 + 0.125, s_tol=1e-12)
    assert line[line.element_names[-1]].length == 0.125 - length


def test_compensated_float_arithmetic_and_serialization():
    origin = xt.CompensatedFloat(2.**30)
    shifted = origin + 2e-10
    assert float(shifted) == float(origin)
    assert shifted.difference_from(origin) == 2e-10
    assert shifted.difference_from(2.**30) == 2e-10
    assert float(shifted - origin) == 2e-10
    assert float(shifted - 2.**30) == 2e-10
    assert (shifted / 2).difference_from(origin / 2) == 1e-10
    assert shifted.difference_from(origin, offset=1e-10) == 1e-10
    restored = xt.CompensatedFloat.from_dict(json.loads(json.dumps(shifted.to_dict())))
    assert restored == shifted
    assert restored.difference_from(origin) == 2e-10
    assert origin.hi == 2.**30 and origin.lo == 0.


@pytest.mark.parametrize('kind', ['int', 'float', 'numpy', 'expression', 'reference', 'sequential'])
def test_normal_placement_values_remain_unchanged(kind):
    env = xt.Environment()
    env.new('a', xt.Marker)
    env.new('b', xt.Marker)
    env['position'] = 2.
    at = {'int': 2, 'float': 2., 'numpy': np.float64(2.),
          'expression': 'position', 'reference': env.ref['position'],
          'sequential': None}[kind]
    place = env.place('b', at=at, anchor='start')
    specification = vars(place).copy()
    composer = xt.Composer(env, components=[xt.Place('a', at=0.), place])
    data_before = composer.to_dict()
    table = composer.resolve_s_positions()
    assert table['s_start', 'b'] == (0. if at is None else 2.)
    assert place.at is at
    assert vars(place) == specification
    assert set(vars(place)) == {'name', 'at', 'from_', 'anchor', 'from_anchor', 'env'}
    assert composer.to_dict() == data_before
    assert not isinstance(data_before['components'][1].get('at'), dict)


@pytest.mark.parametrize('relative', [False, True])
def test_compensated_placement_values_and_line_json_roundtrip(tmp_path, relative):
    env = xt.Environment()
    env.new('a', xt.Marker)
    env.new('b', xt.Marker)
    env.new('c', xt.Marker)
    origin = xt.CompensatedFloat(2.**30)
    line = env.new_line(compose=True, s_tol=1e-12)
    line.place('a', at=origin)
    offset = origin + 2e-10
    line.place('b', at=offset, from_='a' if relative else None)
    # A reference origin allows the same value to exercise both paths.
    line.place('c', at=(origin + origin if relative else origin) + 4e-10)
    specifications = [vars(place).copy() for place in line.composer.components]
    table = line.composer.resolve_s_positions()
    assert table['ds_upstream', 'c'] == 2e-10
    assert line.composer.components[1].at is offset
    assert [vars(place) for place in line.composer.components] == specifications

    path = tmp_path / 'line.json'
    line.to_json(path)
    restored = xt.Line.from_json(path)
    # Reloading finalizes compose-mode lines by default.
    assert restored.mode == 'normal'
    assert restored.composer.components[1].at == offset
    np.testing.assert_array_equal(
        restored.composer.resolve_s_positions().ds_upstream, table.ds_upstream)
    data = json.loads(path.read_text())
    assert data['composer']['components'][1]['at'] == offset.to_dict()
    assert set(data['composer']['components'][1]) == (
        {'name', 'at', 'from_'} if relative else {'name', 'at'})
    line.end_compose()
    np.testing.assert_array_equal(restored.get_table().length, line.get_table().length)
    assert restored[restored.element_names[-2]].length == 2e-10


def test_nested_line_at_compensated_value_is_not_rounded():
    env = xt.Environment()
    env.new('q', xt.Quadrupole, length=0.11)
    env.new('end', xt.Marker)
    cell = env.new_line(components=['q'])
    # This binary offset remains exact through the anchor adjustment.
    offset = 2.**-32
    at = xt.CompensatedFloat(2.**30) + offset
    place = env.place(cell, at=at, anchor='end')
    specification = vars(place).copy()
    composer = xt.Composer(env, s_tol=1e-12, components=[
        place, xt.Place('end', at=xt.CompensatedFloat(2.**30) + 2 * offset),
    ])
    flat = composer.flatten()
    expected_start = at - 0.11
    assert flat.components[0].at == expected_start
    assert place.at is at
    assert vars(place) == specification
    for candidate in [composer, flat, flat.flatten()]:
        assert candidate.resolve_s_positions()['ds_upstream', 'end'] == offset
        line = candidate.build()
        assert line[line.element_names[-2]].length == offset
