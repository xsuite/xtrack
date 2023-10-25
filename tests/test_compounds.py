# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2023.                 #
# ######################################### #
import pytest

import xtrack as xt
import xobjects as xo
from cpymad.madx import Madx

from xtrack.compounds import SlicedCompound, Compound, CompoundContainer
from xtrack.slicing import Strategy, Uniform


def test_compound_container_define_compound():
    sliced_1 = SlicedCompound(['a', 'b'])
    sliced_2 = SlicedCompound(['c', 'd'])
    sliced_3 = SlicedCompound(['e', 'f'])

    container = CompoundContainer(compounds={
        'test_1': sliced_1,
        'test_2': sliced_3,
    })
    container.define_compound('test_2', sliced_2)
    container.define_compound('test_3', sliced_3)

    assert container._compounds == {
        'test_1': sliced_1,
        'test_2': sliced_2,
        'test_3': sliced_3,
    }
    assert container._compound_name_for_element == {
        'a': 'test_1',
        'b': 'test_1',
        'c': 'test_2',
        'd': 'test_2',
        'e': 'test_3',
        'f': 'test_3',
    }


def test_compound_container_remove_compound():
    sliced_1 = SlicedCompound(['a', 'b'])
    sliced_2 = SlicedCompound(['c', 'd'])
    sliced_3 = SlicedCompound(['e', 'f'])

    container = CompoundContainer(compounds={
        'test_1': sliced_1,
        'test_2': sliced_2,
        'test_3': sliced_3,
    })

    container.remove_compound('test_2')

    assert container._compounds == {
        'test_1': sliced_1,
        'test_3': sliced_3,
    }
    assert container._compound_name_for_element == {
        'a': 'test_1',
        'b': 'test_1',
        'e': 'test_3',
        'f': 'test_3',
    }


def test_compound_container_accessors():
    sliced_1 = SlicedCompound(['a', 'b'])
    sliced_2 = SlicedCompound(['c', 'd'])
    sliced_3 = SlicedCompound(['e', 'f'])

    container = CompoundContainer(compounds={
        'test_1': sliced_1,
        'test_2': sliced_2,
        'test_3': sliced_3,
    })

    assert container.compound_for_name('test_1') is sliced_1
    assert container.compound_for_name('test_2') is sliced_2
    assert container.compound_for_name('test_3') is sliced_3

    assert container.compound_name_for_element('a') == 'test_1'
    assert container.compound_name_for_element('b') == 'test_1'
    assert container.compound_name_for_element('c') == 'test_2'
    assert container.compound_name_for_element('d') == 'test_2'
    assert container.compound_name_for_element('e') == 'test_3'
    assert container.compound_name_for_element('f') == 'test_3'

    assert set(container.compound_names) == {'test_1', 'test_2', 'test_3'}


def test_line_compound_container():
    line = xt.Line(
        element_names=['a', 'b', 'c', 'd', 'e', 'f'],
                      # ^test_1   ^c   ^test_2   ^f  # noqa
        elements=[xt.Drift(length=1) for _ in range(6)],
    )

    compound_ab = SlicedCompound({'b', 'a'})
    line.compound_container.define_compound('test_1', compound_ab)

    compound_de = SlicedCompound({'e', 'd'})
    line.compound_container.define_compound('test_2', compound_de)

    assert line.get_compound_mask() == [True, False, True, True, False, True]
    assert line.get_collapsed_names() == ['test_1', 'c', 'test_2', 'f']
    assert line.get_element_compound_names() == [
        'test_1', 'test_1', None, 'test_2', 'test_2', None,
    ]

    assert line.get_compound_for_element('a') == 'test_1'
    assert line.get_compound_for_element('b') == 'test_1'
    assert line.get_compound_for_element('d') == 'test_2'
    assert line.get_compound_for_element('e') == 'test_2'
    assert line.get_compound_for_element('f') is None

    assert line.get_compound_by_name('test_1') is compound_ab
    assert line.get_compound_subsequence('test_1') == ['a', 'b']

    assert line.get_compound_by_name('test_2') is compound_de
    assert line.get_compound_subsequence('test_2') == ['d', 'e']


def test_slicing_preserve_thick_compound_if_unsliced():
    mad = Madx()
    mad.options.rbarc = False
    mad.input(f"""
    ss: sequence, l:=2, refer=entry;
        slice: sbend, at=0, l:=1, angle:=0.1, k0:=0.2, k1=0.1;
        keep: sbend, at=1, l:=1, angle:=0.1, k0:=0.4, k1=0;
    endsequence;
    """)
    mad.beam()
    mad.use(sequence='ss')

    line = xt.Line.from_madx_sequence(
        sequence=mad.sequence.ss,
        deferred_expressions=True,
        allow_thick=True,
    )

    line.slice_thick_elements(slicing_strategies=[
        Strategy(Uniform(2), name='slice'),
        Strategy(None, name='keep'),
    ])

    assert isinstance(line.get_compound_by_name('slice'), SlicedCompound)
    assert line.get_compound_subsequence('slice') == [
        'slice_entry',
        'slice_den',
        'drift_slice..0', 'slice..0',
        'drift_slice..1', 'slice..1',
        'drift_slice..2',
        'slice_dex',
        'slice_exit',
    ]

    assert isinstance(line.get_compound_by_name('keep'), Compound)
    assert line.get_compound_subsequence('keep') == [
        'keep_entry', 'keep_den', 'keep', 'keep_dex', 'keep_exit',
    ]


@pytest.fixture(scope='function')
def line_with_compounds(temp_context_default_func):
    # The fixture `temp_context_default_func` is defined in conftest.py and is
    # needed here as the non-empty context after the tests that invoke
    # this fixture would trigger an error (due to the empty context check).
    # See `conftest.py` for more details.
    elements = [
        ('a', xt.Marker()),         # ┐
        ('b', xt.Marker()),         # │
        ('c', xt.Drift(length=1)),  # ├ test_1
        ('d', xt.Marker()),         # ┘
        ('e', xt.Drift(length=1)),  # ┐
        ('f', xt.Drift(length=1)),  # ┴ test_2
        ('g', xt.Marker()),
    ]
    line = xt.Line(
        element_names=[name for name, _ in elements],
        elements=[element for _, element in elements],
    )

    compound = Compound(entry='a', exit_='d', aperture=['b'], core=['c'])
    line.compound_container.define_compound('test_1', compound)

    sliced_compound = SlicedCompound({'e', 'f'})
    line.compound_container.define_compound('test_2', sliced_compound)

    return line


def test_line_insert_thin_by_index_into_compound(line_with_compounds):
    line = line_with_compounds

    line.insert_element(element=xt.Marker(), index=2, name='c2')

    expected_subsequence = ['a', 'b', 'c2', 'c', 'd']
    assert line.get_compound_subsequence('test_1') == expected_subsequence
    assert 'c2' in line.get_compound_by_name('test_1').core
    assert line.element_names == expected_subsequence + ['e', 'f', 'g']


def test_line_insert_thin_by_index_into_compound_illegal(line_with_compounds):
    line = line_with_compounds

    with pytest.raises(ValueError):
        line.insert_element(element=xt.Marker(), index=1, name='wrong')


def test_line_insert_thin_by_index_into_sliced_compound(line_with_compounds):
    line = line_with_compounds

    line.insert_element(element=xt.Marker(), index=5, name='e2')

    assert line.get_compound_subsequence('test_2') == ['e', 'e2', 'f']


def test_line_insert_thin_by_index_next_to_compound(line_with_compounds):
    line = line_with_compounds

    line.insert_element(element=xt.Marker(), index=6, name='f2')
    assert line.get_compound_subsequence('test_2') == ['e', 'f']

    line.insert_element(element=xt.Marker(), index=4, name='d2')
    assert line.get_compound_subsequence('test_1') == ['a', 'b', 'c', 'd']

    assert line.element_names == ['a', 'b', 'c', 'd', 'd2', 'e', 'f', 'f2', 'g']


def test_line_insert_thin_by_s_next_to_compound(line_with_compounds):
    line = line_with_compounds

    line.insert_element(element=xt.Marker(), at_s=0, name='a0')

    assert line.get_compound_subsequence('test_1') == ['a', 'b', 'c', 'd']
    assert line.element_names == ['a0', 'a', 'b', 'c', 'd', 'e', 'f', 'g']


def test_line_insert_thin_by_s_into_compound_middle(line_with_compounds):
    line = line_with_compounds

    line.insert_element(element=xt.Marker(), at_s=0.5, name='in_c')

    expected_subsequence = ['a', 'b', 'c_u', 'in_c', 'c_d', 'd']
    assert line.get_compound_subsequence('test_1') == expected_subsequence
    assert 'c_u' in line.get_compound_by_name('test_1').core
    assert 'c_d' in line.get_compound_by_name('test_1').core
    assert 'in_c' in line.get_compound_by_name('test_1').core
    assert line.element_names == expected_subsequence + ['e', 'f', 'g']


def test_line_insert_thin_by_s_into_compound_side(line_with_compounds):
    line = line_with_compounds

    line.insert_element(element=xt.Marker(), at_s=1, name='right_of_c')

    expected_subsequence = ['a', 'b', 'c', 'right_of_c', 'd']
    assert line.get_compound_subsequence('test_1') == expected_subsequence
    assert 'right_of_c' in line.get_compound_by_name('test_1').core
    assert line.element_names == expected_subsequence + ['e', 'f', 'g']


def test_line_insert_thin_by_s_into_sliced_compound(line_with_compounds):
    line = line_with_compounds

    line.insert_element(element=xt.Marker(), at_s=2, name='e2')

    assert line.get_compound_subsequence('test_2') == ['e', 'e2', 'f']


def test_line_insert_thick_by_s_into_compound(line_with_compounds):
    line = line_with_compounds

    line.insert_element(element=xt.Drift(length=0.5), at_s=0.25, name='in_c')

    expected_subsequence = ['a', 'b', 'c_u', 'in_c', 'c_d', 'd']
    assert line.get_compound_subsequence('test_1') == expected_subsequence
    assert 'c_u' in line.get_compound_by_name('test_1').core
    assert 'c_d' in line.get_compound_by_name('test_1').core
    assert 'in_c' in line.get_compound_by_name('test_1').core
    assert line.element_names == expected_subsequence + ['e', 'f', 'g']


def test_line_insert_thick_by_s_into_sliced_compound(line_with_compounds):
    line = line_with_compounds

    line.insert_element(element=xt.Drift(length=1), at_s=1.5, name='ef')

    expected_names = ['e_u', 'ef', 'f_d']
    result_names = line.get_compound_subsequence('test_2')
    assert result_names == expected_names

    expected_lengths = [0.5, 1, 0.5]
    result_lengths = [line[name].length for name in result_names]
    assert result_lengths == expected_lengths
