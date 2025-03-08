# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2023.                 #
# ######################################### #
import math

import numpy as np
import pytest

import xobjects as xo
import xtrack as xt
from xtrack.slicing import Strategy, Teapot, Uniform, Custom, Slicer


def test_slicing_uniform():
    # Test for one slice
    slicing_1 = Uniform(1)
    assert slicing_1.element_weights() == [1.0]
    assert slicing_1.drift_weights() == [0.5] * 2
    expected_1 = [w for w in slicing_1.iter_weights(None)]
    result_1 = [(0.5, True), (1.0, False), (0.5, True)]
    assert expected_1 == result_1

    # Test for three slices
    slicing_3 = Uniform(3)
    assert slicing_3.element_weights() == [1/3] * 3
    assert slicing_3.drift_weights() == [0.25] * 4

    elem_info, drift_info = (1./3., False), (0.25, True)
    expected_3 = [
        drift_info, elem_info,
        drift_info, elem_info,
        drift_info, elem_info,
        drift_info,
    ]
    result_3 = [w for w in slicing_3.iter_weights(None)]
    assert expected_3 == result_3

    # Test error handling
    with pytest.raises(ValueError):
        Uniform(0)


def test_slicing_teapot():
    # Test for one slice
    slicing_1 = Teapot(1)
    assert slicing_1.element_weights() == [1.0]
    assert slicing_1.drift_weights() == [0.5] * 2
    expected_1 = [(0.5, True), (1.0, False), (0.5, True)]
    result_1 = [w for w in slicing_1.iter_weights()]
    assert expected_1 == result_1

    # Test for three slices
    slicing_3 = Teapot(3)
    assert slicing_3.element_weights() == [1/3] * 3
    assert slicing_3.drift_weights() == [0.125, 0.375, 0.375, 0.125]

    elem_info = (1./3., False)
    expected_3 = [
        (0.125, True), elem_info,
        (0.375, True), elem_info,
        (0.375, True), elem_info,
        (0.125, True),
    ]
    result_3 = [w for w in slicing_3.iter_weights()]
    assert expected_3 == result_3

    # Test error handling
    with pytest.raises(ValueError):
        Teapot(0)


def test_slicing_teapot_mode_thick():
    # Test for two slices
    slicing_2 = Teapot(2, mode='thick')
    assert slicing_2.drift_weights() == [0.5] * 2
    expected_2 = [(0.5, True), (0.5, True)]
    result_2 = [w for w in slicing_2.iter_weights()]
    assert expected_2 == result_2

    # Test for four slices
    slicing_3 = Teapot(4, mode='thick')
    assert slicing_3.drift_weights() == [0.125, 0.375, 0.375, 0.125]

    expected = [
        (0.125, True),
        (0.375, True),
        (0.375, True),
        (0.125, True),
    ]
    result = [w for w in slicing_3.iter_weights()]
    assert expected == result

    # Test error handling
    with pytest.raises(ValueError):
        Teapot(0)


def test_slicing_custom():
    elem_len_3 = 16
    slicing_3 = Custom(at_s=[0.8, 2, 8], mode='thin')
    expected_dr_3 = [0.8/16, 1.2/16, 6/16, 8/16]
    result_dr_3 = slicing_3.drift_weights(element_length=elem_len_3)
    xo.assert_allclose(expected_dr_3, result_dr_3, atol=1e-30)

    expected_el_3 = [1/3] * 3
    result_el_3 = slicing_3.element_weights(element_length=elem_len_3)
    xo.assert_allclose(expected_el_3, result_el_3, atol=1e-30)

    elem_info = (1/3, False)
    expected_3 = [
        (0.8/16, True), elem_info,
        (1.2/16, True), elem_info,
        (6/16, True), elem_info,
        (8/16, True),
    ]
    result_3 = [w for w in slicing_3.iter_weights(element_length=elem_len_3)]
    assert expected_3 == result_3  # ditto


def test_slicing_custom_thick():
    elem_len_1 = 1.1
    slicing_1 = Custom(at_s=[0.3], mode='thick')
    expected_dr_1 = [0.3 / 1.1, 0.8 / 1.1]
    result_dr_1 = slicing_1.drift_weights(element_length=elem_len_1)
    xo.assert_allclose(expected_dr_1, result_dr_1, atol=1e-30)

    expected_1 = [
        (0.3 / 1.1, True),
        (0.8 / 1.1, True),
    ]
    result_1 = [w for w in slicing_1.iter_weights(element_length=elem_len_1)]
    assert expected_1 == result_1  # for now exact comparison works

    elem_len_3 = 16
    slicing_3 = Custom(at_s=[0.8, 2, 8])
    expected_dr_3 = [0.8/16, 1.2/16, 6/16, 8/16]
    result_dr_3 = slicing_3.drift_weights(element_length=elem_len_3)
    xo.assert_allclose(expected_dr_3, result_dr_3, atol=1e-30)

    expected_3 = [
        (0.8/16, True),
        (1.2/16, True),
        (6/16, True),
        (8/16, True),
    ]
    result_3 = [w for w in slicing_3.iter_weights(element_length=elem_len_3)]
    assert expected_3 == result_3  # ditto


def test_strategy_matching_good_order():
    slicing_strategies = [
        Strategy(slicing=Uniform(1)),
        Strategy(element_type=xt.Drift, slicing=Uniform(2)),
        Strategy(name='some.*', slicing=Uniform(3)),
        Strategy(name='something', slicing=Uniform(4)),
        Strategy(name='something', element_type=xt.Drift, slicing=Uniform(5)),
    ]

    dr = xt.Drift()
    mk = xt.Marker()
    line = xt.Line(elements=[dr, mk])
    slicer = Slicer(slicing_strategies=slicing_strategies, line=line)

    assert slicer._scheme_for_element(element=mk, name='else', line=line).slicing_order == 1
    assert slicer._scheme_for_element(element=dr, name='what', line=line).slicing_order == 2
    assert slicer._scheme_for_element(element=dr, name='somewhat', line=line).slicing_order == 3
    assert slicer._scheme_for_element(element=mk, name='something', line=line).slicing_order == 4
    assert slicer._scheme_for_element(element=dr, name='something', line=line).slicing_order == 5


def test_strategy_matching_confusing_order():
    slicing_strategies = [
        Strategy(slicing=Uniform(1)),
        Strategy(name='something', slicing=Uniform(2)),
        Strategy(name='something', element_type=xt.Drift, slicing=Uniform(3)),
        Strategy(name='some.*', slicing=Uniform(4)),
        Strategy(name='some.*', element_type=xt.Drift, slicing=Uniform(5)),
    ]

    dr = xt.Drift()
    mk = xt.Marker()
    line = xt.Line(elements=[dr, mk])
    slicer = Slicer(slicing_strategies=slicing_strategies, line=line)

    assert slicer._scheme_for_element(element=mk, name='else', line=line).slicing_order == 1
    assert slicer._scheme_for_element(element=mk, name='something', line=line).slicing_order == 4
    assert slicer._scheme_for_element(element=dr, name='something', line=line).slicing_order == 5
    assert slicer._scheme_for_element(element=mk, name='somewhat', line=line).slicing_order == 4
    assert slicer._scheme_for_element(element=dr, name='somewhat', line=line).slicing_order == 5


def test_slicing_strategy_matching():
    elements = [
        ('keep_this', xt.Quadrupole(length=1.0)),
        ('mb10', xt.Bend(length=1.0)),
        ('keep_drifts', xt.Drift(length=1.0)),
        ('mb11', xt.Quadrupole(length=1.0)),
        ('mq10', xt.Quadrupole(length=1.0)),
        ('something', xt.Bend(length=1.0)),
        ('mb20', xt.Bend(length=1.0)),
        ('keep_thin', xt.Multipole(length=1.0)),
        ('mb21', xt.Quadrupole(length=1.0)),
    ]

    slicing_strategies = [
        # Default: one slice
        Strategy(slicing=Uniform(1)),
        # All bends: two slices
        Strategy(slicing=Teapot(2), element_type=xt.Bend),
        # All CFDs: three slices
        Strategy(slicing=Uniform(3), element_type=xt.Quadrupole),
        # If the name starts with mb: five slices (the bend and the cfd 'mb11')
        Strategy(slicing=Teapot(5), name=r'mb1.*'),
        # If the name starts with some: four slices (the bend 'something')
        Strategy(slicing=Uniform(4), name=r'some.*'),
        # Keep the CFD 'keep'
        Strategy(slicing=None, name=r'keep'),
        # mq10: one slice
        Strategy(slicing=Teapot(1), name=r'mq10'),
    ]

    line = xt.Line(
        elements=dict(elements),
        element_names=[name for name, _ in elements],
    )

    line.slice_thick_elements(slicing_strategies)

    # Check that the slices are as expected:
    expected_names = [
        # Kept Quadrupole:
        'keep_this',
        # 5 slices for mb10:
        'mb10_entry',  # Marker
        'mb10..entry_map',
        'drift_mb10..0', 'mb10..0', 'drift_mb10..1', 'mb10..1', 'drift_mb10..2',
        'mb10..2', 'drift_mb10..3', 'mb10..3', 'drift_mb10..4', 'mb10..4',
        'drift_mb10..5',
        'mb10..exit_map',
        'mb10_exit',  # Marker
        # Kept drift:
        'keep_drifts',
        # Five slices for mb11:
        'mb11_entry',  # Marker
        'mb11..entry_map',
        'drift_mb11..0', 'mb11..0', 'drift_mb11..1', 'mb11..1', 'drift_mb11..2',
        'mb11..2', 'drift_mb11..3', 'mb11..3', 'drift_mb11..4', 'mb11..4',
        'drift_mb11..5',
        'mb11..exit_map',
        'mb11_exit',  # Marker
        # One slice for 'mq10':
        'mq10_entry',  # Marker
        'mq10..entry_map',
        'drift_mq10..0', 'mq10..0', 'drift_mq10..1',
        'mq10..exit_map',
        'mq10_exit',  # Marker
        # Four slices for 'something':
        'something_entry',  # Marker
        'something..entry_map',
        'drift_something..0', 'something..0', 'drift_something..1',
        'something..1', 'drift_something..2', 'something..2',
        'drift_something..3', 'something..3', 'drift_something..4',
        'something..exit_map',
        'something_exit',  # Marker
        # Two slices for 'mb20':
        'mb20_entry',  # Marker
        'mb20..entry_map',
        'drift_mb20..0', 'mb20..0', 'drift_mb20..1', 'mb20..1', 'drift_mb20..2',
        'mb20..exit_map',
        'mb20_exit',  # Marker
        # Keep thin:
        'keep_thin',
        # Three slices for 'mb21' (it's a Quadrupole!):
        'mb21_entry',  # Marker
        'mb21..entry_map',
        'drift_mb21..0', 'mb21..0', 'drift_mb21..1', 'mb21..1', 'drift_mb21..2',
        'mb21..2', 'drift_mb21..3',
        'mb21..exit_map',
        'mb21_exit',  # Marker
    ]
    assert line.element_names == expected_names

    # Check types:
    for name, element in line.element_dict.items():
        if name == 'keep_this':
            assert isinstance(element, xt.Quadrupole)
        elif name == 'keep_drifts':
            assert isinstance(element, xt.Drift)
        elif name.startswith('drift_'):
            assert 'DriftSlice' in type(element).__name__
        elif name == 'keep_thin':
            assert isinstance(element, xt.Multipole)
        elif name[-3:-1] == '..':
            assert 'ThinSlice' in type(element).__name__
        else:
            name not in line.element_names

    # Check the right scheme was used:
    def _weights_of_drifts(name):
        return [
            line[nn].weight for nn in line.element_names
            if nn.startswith(f'drift_{name}')
        ]

    # Check markers
    assert isinstance(line['mb10_entry'], xt.Marker)
    assert isinstance(line['mb11_entry'], xt.Marker)
    assert isinstance(line['mq10_entry'], xt.Marker)
    assert isinstance(line['something_entry'], xt.Marker)
    assert isinstance(line['mb20_entry'], xt.Marker)
    assert isinstance(line['mb21_entry'], xt.Marker)
    assert isinstance(line['mb10_exit'], xt.Marker)
    assert isinstance(line['mb11_exit'], xt.Marker)
    assert isinstance(line['mq10_exit'], xt.Marker)
    assert isinstance(line['something_exit'], xt.Marker)
    assert isinstance(line['mb20_exit'], xt.Marker)
    assert isinstance(line['mb21_exit'], xt.Marker)

    # Teapot
    expected_mb20_drift_lens = [1/6, 2/3, 1/6]
    assert _weights_of_drifts('mb20') == expected_mb20_drift_lens

    expected_mb10_drift_lens = [1/12, 5/24, 5/24, 5/24, 5/24, 1/12]
    assert _weights_of_drifts('mb10') == expected_mb10_drift_lens

    expected_mb11_drift_lens = [1/12, 5/24, 5/24, 5/24, 5/24, 1/12]
    assert _weights_of_drifts('mb11') == expected_mb11_drift_lens

    # Uniform
    expected_mq10_drift_lens = [1/2] * 2
    assert _weights_of_drifts('mq10') == expected_mq10_drift_lens

    expected_mb21_drift_lens = [1/4] * 4
    assert _weights_of_drifts('mb21') == expected_mb21_drift_lens

    expected_something_drift_lens = [1/5] * 5
    assert _weights_of_drifts('something') == expected_something_drift_lens

    tt = line.get_table()

    # Test accessing compound elements
    assert np.all(tt.rows['mb10_entry':'mb10_exit'].name == [
        'mb10_entry', 'mb10..entry_map',
        'drift_mb10..0', 'mb10..0', 'drift_mb10..1', 'mb10..1',
        'drift_mb10..2', 'mb10..2', 'drift_mb10..3', 'mb10..3', 'drift_mb10..4',
        'mb10..4', 'drift_mb10..5',
        'mb10..exit_map', 'mb10_exit',
    ])
    assert np.all(tt.rows['mb11_entry':'mb11_exit'].name == [
        'mb11_entry', 'mb11..entry_map',
        'drift_mb11..0', 'mb11..0', 'drift_mb11..1', 'mb11..1',
        'drift_mb11..2', 'mb11..2', 'drift_mb11..3', 'mb11..3', 'drift_mb11..4',
        'mb11..4', 'drift_mb11..5',
        'mb11..exit_map', 'mb11_exit',
    ])
    assert np.all(tt.rows['mq10_entry':'mq10_exit'].name == [
        'mq10_entry', 'mq10..entry_map',
        'drift_mq10..0', 'mq10..0', 'drift_mq10..1',
        'mq10..exit_map', 'mq10_exit',
    ])
    assert np.all(tt.rows['something_entry':'something_exit'].name == [
        'something_entry', 'something..entry_map',
        'drift_something..0', 'something..0',
        'drift_something..1', 'something..1', 'drift_something..2',
        'something..2', 'drift_something..3', 'something..3',
        'drift_something..4',
        'something..exit_map', 'something_exit',
    ])
    assert np.all(tt.rows['mb20_entry':'mb20_exit'].name == [
        'mb20_entry', 'mb20..entry_map',
        'drift_mb20..0', 'mb20..0', 'drift_mb20..1', 'mb20..1',
        'drift_mb20..2',
        'mb20..exit_map', 'mb20_exit',
    ])
    assert np.all(tt.rows['mb21_entry':'mb21_exit'].name == [
        'mb21_entry', 'mb21..entry_map',
        'drift_mb21..0', 'mb21..0', 'drift_mb21..1', 'mb21..1',
        'drift_mb21..2', 'mb21..2', 'drift_mb21..3',
        'mb21..exit_map', 'mb21_exit',
    ])


def test_slicing_thick_bend_simple():

    additional_kwargs = {}
    additional_kwargs['k1'] = 0.2

    bend = xt.Bend(
        length=3.0,
        k0=0.1,
        h=0.2,
        **additional_kwargs
    )
    line = xt.Line(elements=[bend], element_names=['bend'])
    line.slice_thick_elements([Strategy(slicing=Teapot(2))])

    assert len(line) == 9  # 2 markers + 2 edges + 2 slices + 3 drifts

    assert line['drift_bend..0'].weight == 1/6
    assert line['drift_bend..1'].weight == 2/3
    assert line['drift_bend..2'].weight == 1/6

    bend0, bend1 = line['bend..0'], line['bend..1']
    assert bend0.weight == bend1.weight == 0.5

    # Make sure the order and the inverse factorial make sense:
    _fact = math.factorial
    xo.assert_allclose(_fact(bend0._parent.order) * bend0._parent.inv_factorial_order, 1, atol=1e-16)
    xo.assert_allclose(_fact(bend1._parent.order) * bend0._parent.inv_factorial_order, 1, atol=1e-16)


def test_slicing_thick_bend_into_thick_bends_simple():

    additional_kwargs = {}
    additional_kwargs['k0'] = 0.1
    additional_kwargs['h'] = 0.2
    additional_kwargs['k1'] = 0.2

    bend = xt.Bend(
        length=3.0,
        knl=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        ksl=[0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
        **additional_kwargs
    )
    line = xt.Line(elements=[bend], element_names=['bend'])
    line.slice_thick_elements([Strategy(slicing=Uniform(2, mode='thick'))])

    assert len(line) == 6  # 2 markers + 2 edges + 2 thick slices

    bend0, bend1 = line['bend..0'], line['bend..1']
    assert bend0.weight == bend1.weight == 0.5

    assert bend0._parent.k0 == bend1._parent.k0 == 0.1
    assert bend0._parent.h == bend1._parent.h == 0.2
    assert bend0._parent.k1 == bend1._parent.k1 == 0.2

    expected_knl = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    xo.assert_allclose(bend0._parent.knl, expected_knl, atol=1e-16)
    xo.assert_allclose(bend1._parent.knl, expected_knl, atol=1e-16)

    expected_ksl = np.array([0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
    xo.assert_allclose(bend0._parent.ksl, expected_ksl, atol=1e-16)
    xo.assert_allclose(bend1._parent.ksl, expected_ksl, atol=1e-16)

    # Make sure the order and the inverse factorial make sense:
    _fact = math.factorial
    xo.assert_allclose(_fact(bend0._parent.order) * bend0._parent.inv_factorial_order, 1, atol=1e-16)
    xo.assert_allclose(_fact(bend1._parent.order) * bend0._parent.inv_factorial_order, 1, atol=1e-16)


def test_slicing_xdeps_consistency():
    num_elements = 50000
    num_slices = 1

    line = xt.Line(
        elements=[xt.Bend(k0=1, length=100)] * num_elements,
        element_names=[f'bend{ii}' for ii in range(num_elements)],
    )
    line._init_var_management()

    for ii in range(num_elements):
        line.vars[f'k{ii}'] = 1
        line.element_refs[f'bend{ii}'].k0 = line.vars[f'k{ii}']

    sgy = xt.slicing.Strategy(
        element_type=xt.Bend,
        slicing=xt.slicing.Uniform(num_slices),
    )
    line.slice_thick_elements([sgy])
    assert len(line.to_dict()['_var_manager']) == num_elements * num_slices

def test_slice_twice():
    env = xt.Environment()
    line = env.new_line(components=[
        env.new('el', xt.Bend, length=4, k1=0.5, h=0.1, k0=0.8, knl=[0, 0, 0.03]),
    ])

    line.slice_thick_elements(
        slicing_strategies=[
            xt.Strategy(slicing=xt.Teapot(4, mode='thick'), element_type=xt.Bend),
        ])

    tt_first_slice = line.get_table(attr=True)

    line.slice_thick_elements(
        slicing_strategies=[
            xt.Strategy(slicing=xt.Teapot(4, mode='thin'), element_type=xt.ThickSliceBend),
        ])

    tt_second_slice = line.get_table(attr=True)

    assert np.all(tt_first_slice.name == np.array(
        ['el_entry', 'el..entry_map', 'el..0', 'el..1', 'el..2', 'el..3',
        'el..exit_map', 'el_exit', '_end_point']))

    assert np.all(tt_first_slice.element_type == np.array(
        ['Marker', 'ThinSliceBendEntry', 'ThickSliceBend', 'ThickSliceBend',
        'ThickSliceBend', 'ThickSliceBend', 'ThinSliceBendExit', 'Marker',
        '']))

    # Check first table

    xo.assert_allclose(tt_first_slice.angle_rad, np.array(
        [0.  , 0.  , 0.05, 0.15, 0.15, 0.05, 0.  , 0.  , 0.  ]), rtol=0, atol=1e-12)
    xo.assert_allclose(np.sum(tt_first_slice.angle_rad), 0.4, rtol=0, atol=1e-12)

    xo.assert_allclose(tt_first_slice.k0l, np.array(
        [0. , 0. , 0.4, 1.2, 1.2, 0.4, 0. , 0. , 0. ]), rtol=0, atol=1e-12)
    xo.assert_allclose(np.sum(tt_first_slice.k0l), 3.2, rtol=0, atol=1e-12)

    xo.assert_allclose(tt_first_slice.k1l, np.array(
        [0.  , 0.  , 0.25, 0.75, 0.75, 0.25, 0.  , 0.  , 0.  ]), rtol=0, atol=1e-12)
    xo.assert_allclose(np.sum(tt_first_slice.k1l), 2, rtol=0, atol=1e-12)

    xo.assert_allclose(tt_first_slice.k2l, np.array(
        [0.     , 0.     , 0.00375, 0.01125, 0.01125, 0.00375, 0. , 0., 0.]),
        rtol=0, atol=1e-12)
    xo.assert_allclose(np.sum(tt_first_slice.k2l), 0.03, rtol=0, atol=1e-12)

    # Check second table

    assert np.all(tt_second_slice.name == np.array(
        ['el_entry', 'el..entry_map', 'el..0_entry', 'drift_el..0..0',
        'el..0..0', 'drift_el..0..1', 'el..0..1', 'drift_el..0..2',
        'el..0..2', 'drift_el..0..3', 'el..0..3', 'drift_el..0..4',
        'el..0_exit', 'el..1_entry', 'drift_el..1..0', 'el..1..0',
        'drift_el..1..1', 'el..1..1', 'drift_el..1..2', 'el..1..2',
        'drift_el..1..3', 'el..1..3', 'drift_el..1..4', 'el..1_exit',
        'el..2_entry', 'drift_el..2..0', 'el..2..0', 'drift_el..2..1',
        'el..2..1', 'drift_el..2..2', 'el..2..2', 'drift_el..2..3',
        'el..2..3', 'drift_el..2..4', 'el..2_exit', 'el..3_entry',
        'drift_el..3..0', 'el..3..0', 'drift_el..3..1', 'el..3..1',
        'drift_el..3..2', 'el..3..2', 'drift_el..3..3', 'el..3..3',
        'drift_el..3..4', 'el..3_exit', 'el..exit_map', 'el_exit',
        '_end_point']))

    assert np.all(tt_second_slice.element_type == np.array(
        ['Marker', 'ThinSliceBendEntry', 'Marker', 'DriftSliceBend',
        'ThinSliceBend', 'DriftSliceBend', 'ThinSliceBend',
        'DriftSliceBend', 'ThinSliceBend', 'DriftSliceBend',
        'ThinSliceBend', 'DriftSliceBend', 'Marker', 'Marker',
        'DriftSliceBend', 'ThinSliceBend', 'DriftSliceBend',
        'ThinSliceBend', 'DriftSliceBend', 'ThinSliceBend',
        'DriftSliceBend', 'ThinSliceBend', 'DriftSliceBend', 'Marker',
        'Marker', 'DriftSliceBend', 'ThinSliceBend', 'DriftSliceBend',
        'ThinSliceBend', 'DriftSliceBend', 'ThinSliceBend',
        'DriftSliceBend', 'ThinSliceBend', 'DriftSliceBend', 'Marker',
        'Marker', 'DriftSliceBend', 'ThinSliceBend', 'DriftSliceBend',
        'ThinSliceBend', 'DriftSliceBend', 'ThinSliceBend',
        'DriftSliceBend', 'ThinSliceBend', 'DriftSliceBend', 'Marker',
        'ThinSliceBendExit', 'Marker', ''
        ]))

    xo.assert_allclose(tt_second_slice.angle_rad, np.array(
        [0.    , 0.    , 0.    , 0.    , 0.0125, 0.    , 0.0125, 0.    ,
            0.0125, 0.    , 0.0125, 0.    , 0.    , 0.    , 0.    , 0.0375,
            0.    , 0.0375, 0.    , 0.0375, 0.    , 0.0375, 0.    , 0.    ,
            0.    , 0.    , 0.0375, 0.    , 0.0375, 0.    , 0.0375, 0.    ,
            0.0375, 0.    , 0.    , 0.    , 0.    , 0.0125, 0.    , 0.0125,
            0.    , 0.0125, 0.    , 0.0125, 0.    , 0.    , 0.    , 0.    ,
            0.    ]), rtol=0, atol=1e-12)
    xo.assert_allclose(np.sum(tt_second_slice.angle_rad), 0.4, rtol=0, atol=1e-12)

    xo.assert_allclose(tt_second_slice.k0l, np.array(
        [0. , 0. , 0. , 0. , 0.1, 0. , 0.1, 0. , 0.1, 0. , 0.1, 0. , 0. ,
        0. , 0. , 0.3, 0. , 0.3, 0. , 0.3, 0. , 0.3, 0. , 0. , 0. , 0. ,
        0.3, 0. , 0.3, 0. , 0.3, 0. , 0.3, 0. , 0. , 0. , 0. , 0.1, 0. ,
        0.1, 0. , 0.1, 0. , 0.1, 0. , 0. , 0. , 0. , 0. ]), rtol=0, atol=1e-12)
    xo.assert_allclose(np.sum(tt_second_slice.k0l), 3.2, rtol=0, atol=1e-12)

    xo.assert_allclose(tt_second_slice.k1l, np.array(
        [0.    , 0.    , 0.    , 0.    , 0.0625, 0.    , 0.0625, 0.    ,
        0.0625, 0.    , 0.0625, 0.    , 0.    , 0.    , 0.    , 0.1875,
        0.    , 0.1875, 0.    , 0.1875, 0.    , 0.1875, 0.    , 0.    ,
        0.    , 0.    , 0.1875, 0.    , 0.1875, 0.    , 0.1875, 0.    ,
        0.1875, 0.    , 0.    , 0.    , 0.    , 0.0625, 0.    , 0.0625,
        0.    , 0.0625, 0.    , 0.0625, 0.    , 0.    , 0.    , 0.    ,
        0.    ]), rtol=0, atol=1e-12)
    xo.assert_allclose(np.sum(tt_second_slice.k1l), 2, rtol=0, atol=1e-12)

    xo.assert_allclose(tt_second_slice.k2l, np.array(
        [0.       , 0.       , 0.       , 0.       , 0.0009375, 0.       ,
        0.0009375, 0.       , 0.0009375, 0.       , 0.0009375, 0.       ,
        0.       , 0.       , 0.       , 0.0028125, 0.       , 0.0028125,
        0.       , 0.0028125, 0.       , 0.0028125, 0.       , 0.       ,
        0.       , 0.       , 0.0028125, 0.       , 0.0028125, 0.       ,
        0.0028125, 0.       , 0.0028125, 0.       , 0.       , 0.       ,
        0.       , 0.0009375, 0.       , 0.0009375, 0.       , 0.0009375,
        0.       , 0.0009375, 0.       , 0.       , 0.       , 0.       ,
        0.       ]),
        rtol=0, atol=1e-12)
    xo.assert_allclose(np.sum(tt_second_slice.k2l), 0.03, rtol=0, atol=1e-12)

def test_slice_repeated_elements():

    env = xt.Environment(
        particle_ref=xt.Particles(p0c=7000e9, x=1e-3, px=1e-3, y=1e-3, py=1e-3))

    line0 = env.new_line(
        components=[
            env.new('b0', 'Bend', length=1.0, anchor='start', at=5.0),
            env.new('q0', 'Quadrupole', length=2.0, at=20.0),
            env.new('ql', 'Quadrupole', length=2.0, at=-10.0, from_='q0'),
            env.new('qr', 'Quadrupole', length=2.0, at=10.0, from_='q0'),
            env.new('mk1', 'Marker', at=40),
            env.new('mk2', 'Marker'),
            env.new('mk3', 'Marker'),
            env.place('q0'),
            env.place('b0'),
            env.new('end', 'Marker', at=50.),
        ])
    tt0 = line0.get_table()
    tt0.show(cols=['name', 's_start', 's_end', 's_center'])

    line = line0.copy()
    line.slice_thick_elements(
        slicing_strategies=[
            xt.Strategy(None),
            xt.Strategy(slicing=xt.Teapot(2), name=r'q0.*'),
            xt.Strategy(slicing=xt.Teapot(3), name=r'b0.*'),
        ]
    )
    tt = line.get_table()
    tt.show(cols=['name', 's_start', 's_end', 's_center'])

    assert np.all(tt.name == np.array(
        ['drift_1', 'b0_entry::0', 'b0..entry_map', 'drift_b0..0', 'b0..0',
       'drift_b0..1', 'b0..1', 'drift_b0..2', 'b0..2', 'drift_b0..3',
       'b0..exit_map', 'b0_exit::0', 'drift_2', 'ql', 'drift_3',
       'q0_entry::0', 'q0..entry_map', 'drift_q0..0', 'q0..0',
       'drift_q0..1', 'q0..1', 'drift_q0..2', 'q0..exit_map',
       'q0_exit::0', 'drift_4', 'qr', 'drift_5', 'mk1', 'mk2', 'mk3',
       'q0_entry::1', 'q0..entry_map_0', 'drift_q0..3', 'q0..2',
       'drift_q0..4', 'q0..3', 'drift_q0..5', 'q0..exit_map_0',
       'q0_exit::1', 'b0_entry::1', 'b0..entry_map_0', 'drift_b0..4',
       'b0..3', 'drift_b0..5', 'b0..4', 'drift_b0..6', 'b0..5',
       'drift_b0..7', 'b0..exit_map_0', 'b0_exit::1', 'drift_6', 'end',
       '_end_point']))
    xo.assert_allclose(tt.s_center, np.array(
        [ 2.5       ,  5.        ,  5.        ,  5.0625    ,  5.125     ,
        5.3125    ,  5.5       ,  5.6875    ,  5.875     ,  5.9375    ,
        6.        ,  6.        ,  7.5       , 10.        , 15.        ,
       19.        , 19.        , 19.16666667, 19.33333333, 20.        ,
       20.66666667, 20.83333333, 21.        , 21.        , 25.        ,
       30.        , 35.5       , 40.        , 40.        , 40.        ,
       40.        , 40.        , 40.16666667, 40.33333333, 41.        ,
       41.66666667, 41.83333333, 42.        , 42.        , 42.        ,
       42.        , 42.0625    , 42.125     , 42.3125    , 42.5       ,
       42.6875    , 42.875     , 42.9375    , 43.        , 43.        ,
       46.5       , 50.        , 50.        ]),
        rtol=0., atol=1e-8)

    line = line0.copy()
    line.slice_thick_elements(
        slicing_strategies=[
            xt.Strategy(None),
            xt.Strategy(slicing=xt.Teapot(2), name=r'q0::0'),
            xt.Strategy(slicing=xt.Teapot(3), name=r'b0::1'),
        ]
    )
    tt = line.get_table()
    tt.show(cols=['name', 's_start', 's_end', 's_center'])

    assert np.all(tt.name == np.array(
        ['drift_1', 'b0', 'drift_2', 'ql', 'drift_3', 'q0_entry',
       'q0..entry_map', 'drift_q0..0', 'q0..0', 'drift_q0..1', 'q0..1',
       'drift_q0..2', 'q0..exit_map', 'q0_exit', 'drift_4', 'qr',
       'drift_5', 'mk1', 'mk2', 'mk3', 'q0', 'b0_entry', 'b0..entry_map',
       'drift_b0..0', 'b0..0', 'drift_b0..1', 'b0..1', 'drift_b0..2',
       'b0..2', 'drift_b0..3', 'b0..exit_map', 'b0_exit', 'drift_6',
       'end', '_end_point']))

    xo.assert_allclose(tt.s_center, np.array(
        [ 2.5       ,  5.5       ,  7.5       , 10.        , 15.        ,
          19.        , 19.        , 19.16666667, 19.33333333, 20.        ,
          20.66666667, 20.83333333, 21.        , 21.        , 25.        ,
          30.        , 35.5       , 40.        , 40.        , 40.        ,
          41.        , 42.        , 42.        , 42.0625    , 42.125     ,
          42.3125    , 42.5       , 42.6875    , 42.875     , 42.9375    ,
          43.        , 43.        , 46.5       , 50.        , 50.        ]),
        rtol=0., atol=1e-8)

    line = line0.copy()
    line.slice_thick_elements(
        slicing_strategies=[
            xt.Strategy(None),
            xt.Strategy(slicing=xt.Teapot(2, mode='thick'), name=r'q0.*'),
            xt.Strategy(slicing=xt.Teapot(3, mode='thick'), name=r'b0.*'),
        ],
    )

    tt = line.get_table()
    tt.show(cols=['name', 's_start', 's_end', 's_center'])

    assert np.all(tt.name == np.array(
        ['drift_1', 'b0_entry::0', 'b0..entry_map', 'b0..0', 'b0..1',
       'b0..2', 'b0..exit_map', 'b0_exit::0', 'drift_2', 'ql', 'drift_3',
       'q0_entry::0', 'q0..entry_map', 'q0..0', 'q0..1', 'q0..exit_map',
       'q0_exit::0', 'drift_4', 'qr', 'drift_5', 'mk1', 'mk2', 'mk3',
       'q0_entry::1', 'q0..entry_map_0', 'q0..2', 'q0..3',
       'q0..exit_map_0', 'q0_exit::1', 'b0_entry::1', 'b0..entry_map_0',
       'b0..3', 'b0..4', 'b0..5', 'b0..exit_map_0', 'b0_exit::1',
       'drift_6', 'end', '_end_point']))
    xo.assert_allclose(tt.s_center, np.array(
        [ 2.5       ,  5.        ,  5.        ,  5.08333333,  5.5       ,
          5.91666667,  6.        ,  6.        ,  7.5       , 10.        ,
          15.        , 19.        , 19.        , 19.5       , 20.5       ,
          21.        , 21.        , 25.        , 30.        , 35.5       ,
          40.        , 40.        , 40.        , 40.        , 40.        ,
          40.5       , 41.5       , 42.        , 42.        , 42.        ,
          42.        , 42.08333333, 42.5       , 42.91666667, 43.        ,
          43.        , 46.5       , 50.        , 50.        ]),
        rtol=0., atol=1e-8)

    line = line0.copy()
    line.cut_at_s([20.1, 20.2, 41.7, 41.8, 5.5])

    tt = line.get_table()
    tt.show(cols=['name', 's_start', 's_end', 's_center'])

    assert np.all(tt.name == np.array(
        ['drift_1', 'b0_entry', 'b0..entry_map', 'b0..0', 'b0..1',
       'b0..exit_map', 'b0_exit', 'drift_2', 'ql', 'drift_3',
       'q0_entry::0', 'q0..entry_map', 'q0..0', 'q0..1', 'q0..2',
       'q0..exit_map', 'q0_exit::0', 'drift_4', 'qr', 'drift_5', 'mk1',
       'mk2', 'mk3', 'q0_entry::1', 'q0..entry_map_0', 'q0..3', 'q0..4',
       'q0..5', 'q0..exit_map_0', 'q0_exit::1', 'b0', 'drift_6', 'end',
       '_end_point']))

    xo.assert_allclose(tt.s_center, np.array(
        [ 2.5 ,  5.  ,  5.  ,  5.25,  5.75,  6.  ,  6.  ,  7.5 , 10.  ,
       15.  , 19.  , 19.  , 19.55, 20.15, 20.6 , 21.  , 21.  , 25.  ,
       30.  , 35.5 , 40.  , 40.  , 40.  , 40.  , 40.  , 40.85, 41.75,
       41.9 , 42.  , 42.  , 42.5 , 46.5 , 50.  , 50.  ]),
        rtol=0., atol=1e-8)
