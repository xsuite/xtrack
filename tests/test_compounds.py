# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2023.                 #
# ######################################### #

import xtrack as xt
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
    ! Make the sequence a bit longer to accommodate rbends
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
