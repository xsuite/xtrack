# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import pathlib
import pickle
import math

import numpy as np
import pytest

import xobjects as xo
import xpart as xp
import xtrack as xt
from xobjects.test_helpers import for_all_test_contexts
from xtrack import Line, Node, Multipole

test_data_folder = pathlib.Path(
            __file__).parent.joinpath('../test_data').absolute()


def test_simplification_methods():

    line = xt.Line(
        elements=([xt.Drift(length=0)] # Start line marker
                    + [xt.Drift(length=1) for _ in range(5)]
                    + [xt.Drift(length=0)] # End line marker
            )
        )

    # Test merging of drifts
    line.insert_element(element=xt.Cavity(), name='cav', at_s=3.3)
    assert isinstance(line['e4..0'], xt.DriftSlice)
    line._replace_with_equivalent_elements()
    assert isinstance(line['e4..0'], xt.Drift)
    line.merge_consecutive_drifts(inplace=True)
    assert len(line.element_names) == 3
    assert line.get_length() == line.get_s_elements(mode='downstream')[-1] == 5
    xo.assert_allclose(line[line.element_names[0]].length, 3.3, rtol=0, atol=1e-12)
    assert isinstance(line[line.element_names[1]], xt.Cavity)
    xo.assert_allclose(line[line.element_names[2]].length, 1.7, rtol=0, atol=1e-12)

    # Test merging of drifts, while keeping one
    line.insert_element(element=xt.Drift(length=1), name='drift1', at_s=1.2)
    line.insert_element(element=xt.Drift(length=1), name='drift2', at_s=2.2)
    line._replace_with_equivalent_elements()
    line.merge_consecutive_drifts(inplace=True, keep=['drift2'])
    assert len(line.element_names) == 5
    assert 'drift2' in line.element_names
    assert 'drift1' not in line.element_names
    line._replace_with_equivalent_elements()
    line.merge_consecutive_drifts(inplace=True)

    # Test removing of zero-length drifts
    line.insert_element(element=xt.Drift(length=0), name='dzero1', at_s=3.3)
    line.insert_element(element=xt.Drift(length=0), name='dzero2', at_s=3.3)
    assert len(line.element_names) == 5
    line._replace_with_equivalent_elements()
    line.remove_zero_length_drifts(inplace=True, keep='dzero2')
    assert len(line.element_names) == 4
    assert 'dzero2' in line.element_names
    assert 'dzero1' not in line.element_names
    line._replace_with_equivalent_elements()
    line.remove_zero_length_drifts(inplace=True)

    # Test merging of multipoles
    line._var_management = None
    line.insert_element(element=xt.Multipole(knl=[1, 0, 3], ksl=[0, 20, 0]), name='m1', at_s=3.3)
    line.insert_element(element=xt.Multipole(knl=[4, 2], ksl=[10, 40]), name='m2', at_s=3.3)
    line.insert_element(element=xt.Multipole(knl=[0, 3, 8], ksl=[2, 0, 17]), name='m3', at_s=3.3)
    line.insert_element(element=xt.Multipole(knl=[2, 0, 0], ksl=[40]), name='m4', at_s=3.3)
    assert len(line.element_names) == 7
    line._replace_with_equivalent_elements()
    line.merge_consecutive_multipoles(inplace=True, keep='m3')
    assert len(line.element_names) == 6
    assert 'm3' in line.element_names
    # We merged the first two multipoles
    joined_mult = [ name for name in line.element_names if 'm1' in name]
    assert len(joined_mult) == 1
    joined_mult = joined_mult[0]
    assert 'm2' in joined_mult
    xo.assert_allclose(line[joined_mult].knl, [5,2,3], rtol=0, atol=1e-15)
    xo.assert_allclose(line[joined_mult].ksl, [10,60,0], rtol=0, atol=1e-15)
    # Merging all
    line._replace_with_equivalent_elements()
    line.merge_consecutive_multipoles(inplace=True)
    assert len(line.element_names) == 4
    xo.assert_allclose(line[line.element_names[1]].knl, [7,5,11], rtol=0, atol=1e-15)
    xo.assert_allclose(line[line.element_names[1]].ksl, [52,60,17], rtol=0, atol=1e-15)

    # Test removing inactive multipoles
    line.insert_element(element=xt.Multipole(knl=[0, 8, 1], ksl=[0, 20, 30]), name='m5', at_s=3.3)
    line.insert_element(element=xt.Multipole(knl=[2, 0, 3], ksl=[10, 34, 15]), name='m6', at_s=3.3)
    line.remove_inactive_multipoles(inplace=True)
    assert len(line.element_names) == 6
    line['m5'].knl[:] = 0
    line['m5'].ksl[:] = 0
    line['m6'].knl[:] = 0
    line['m6'].ksl[:] = 0
    line._replace_with_equivalent_elements()
    line.remove_inactive_multipoles(inplace=True, keep='m5')
    assert len(line.element_names) == 5
    assert 'm5' in line.element_names
    assert 'm6' not in line.element_names

    # Test removing markers
    line.insert_element(element=xt.Marker(), name='marker1', at_s=3.3)
    line.insert_element(element=xt.Marker(), name='marker2', at_s=3.3)
    assert 'marker1' in line.element_names
    assert 'marker2' in line.element_names
    line._replace_with_equivalent_elements()
    line.remove_markers(keep='marker2')
    assert 'marker1' not in line.element_names
    assert 'marker2' in line.element_names
    line.insert_element(element=xt.Marker(), name='marker4', at_s=3.3)
    line.insert_element(element=xt.Marker(), name='marker3', at_s=3.3)
    assert 'marker2' in line.element_names
    assert 'marker3' in line.element_names
    assert 'marker4' in line.element_names
    line._replace_with_equivalent_elements()
    line.remove_markers()
    assert 'marker2' not in line.element_names
    assert 'marker3' not in line.element_names
    assert 'marker4' not in line.element_names


def test_remove_redundant_apertures():

    # Lattice:
    # D1-A1-M-D2 D3-A2-M-D4 D5-A3-M-D6 D7-A4-M-D8 D9-A5-M-D10
    elements = []
    for _ in range(5):
        elements += [
            xt.Drift(length=0.6),
            xt.LimitRect(min_x=-0.3, max_x=0.3,min_y=-0.3, max_y=0.3),
            xt.Marker(),
            xt.Drift(length=0.4)
        ]
    line = xt.Line(elements=elements)
    original_line = line.copy()

    # Test removing all consecutive middle apertures
    assert len(line.element_names) == 20
    all_aper = [nn for nn in line.element_names if xt._is_aperture(line[nn], line)]
    all_aper_pos = [line.get_s_position(ap) for ap in all_aper]
    line.remove_redundant_apertures()
    line.remove_markers()
    line.merge_consecutive_drifts()
    # The lattice is now D1-A1-DD-A5-D10
    assert len(line.element_names) == 5
    # Verify that only the first and last aperture are kept
    new_aper = [nn for nn in line.element_names if xt._is_aperture(line[nn], line)]
    assert new_aper == [all_aper[0], all_aper[-1]]
    new_aper_pos = [line.get_s_position(ap) for ap in new_aper]
    assert new_aper_pos == [all_aper_pos[0], all_aper_pos[-1]]

    # Test removing all consecutive middle apertures, but
    # keep the 4th one (and hence also the 5th)
    line = original_line.copy()
    assert len(line.element_names) == 20
    all_aper = [nn for nn in line.element_names if xt._is_aperture(line[nn], line)]
    all_aper_pos = [line.get_s_position(ap) for ap in all_aper]
    line.remove_redundant_apertures(keep=all_aper[3])
    line.remove_markers()
    line.merge_consecutive_drifts()
    # The lattice is now D1-A1-DD-A4-DD-A5-D10
    assert len(line.element_names) == 7
    # Verify that only the first, fourth, and last aperture are kept
    new_aper = [nn for nn in line.element_names if xt._is_aperture(line[nn], line)]
    assert new_aper == [all_aper[0], all_aper[3], all_aper[-1]]
    new_aper_pos = [line.get_s_position(ap) for ap in new_aper]
    assert new_aper_pos == [all_aper_pos[0], all_aper_pos[3], all_aper_pos[-1]]

    # Test removing all consecutive middle apertures, but
    # the 9th Drift needs to have an aperture. This should
    # give the same result as above
    line = original_line.copy()
    assert len(line.element_names) == 20
    all_aper = [nn for nn in line.element_names if xt._is_aperture(line[nn], line)]
    all_aper_pos = [line.get_s_position(ap) for ap in all_aper]
    all_drifts = [nn for nn in line.element_names if xt._is_drift(line[nn], line)]
    line.remove_redundant_apertures(drifts_that_need_aperture=all_drifts[8])
    line.remove_markers()
    line.merge_consecutive_drifts()
    # The lattice is now D1-A1-DD-A4-DD-A5-D10
    assert len(line.element_names) == 7
    # Verify that only the first, fourth, and last aperture are kept
    new_aper = [nn for nn in line.element_names if xt._is_aperture(line[nn], line)]
    assert new_aper == [all_aper[0], all_aper[3], all_aper[-1]]
    new_aper_pos = [line.get_s_position(ap) for ap in new_aper]
    assert new_aper_pos == [all_aper_pos[0], all_aper_pos[3], all_aper_pos[-1]]

    # All apertures are different, none should be removed
    elements = []
    for i in range(5):
        elements += [
            xt.Drift(length=0.6),
            xt.LimitRect(min_x=-0.3+i*0.01, max_x=0.3,min_y=-0.3, max_y=0.3),
            xt.Marker(),
            xt.Drift(length=0.4)
        ]
    line = xt.Line(elements=elements)
    original_line = line.copy()
    line.remove_redundant_apertures()
    assert xt._lines_equal(line, original_line)

def test_redundant_apertures():
    sequence = [
        ('a1', xt.LimitRect(min_x=-0.3, max_x=0.3, min_y=-0.3, max_y=0.3)),
        ('d0', xt.Drift(length=0.6)),
        ('a1', xt.LimitRect(min_x=-0.3, max_x=0.3, min_y=-0.3, max_y=0.3)),
        ('d1', xt.Drift(length=0.4)),
        ('a2', xt.Replica('a1')),
        ('m2', xt.Marker()),
        ('d2..1', xt.Drift(length=0.2)),
        ('d2..2', xt.Drift(length=0.2)),
        ('a3', xt.LimitRect(min_x=-0.3, max_x=0.3, min_y=-0.3, max_y=0.3)),
        ('d3', xt.Drift(length=0.4)),
    ]
    line = xt.Line(elements=dict(sequence), element_names=[n for n, _ in sequence])

    line.remove_redundant_apertures()

    expected_names = ['d0', 'a1', 'd1', 'm2', 'd2..1', 'd2..2', 'a3', 'd3']
    assert line.element_names == expected_names

def test_insert():

    line0 = xt.Line(
        elements = [xt.Drift(length=1) for _ in range(5)]
    )

    line = line0.copy()
    assert np.all(np.array([0,1,2,3,4]) == np.array(line.get_s_elements()))
    assert np.all(np.array([0,1,2,3,4]) == np.array(line.get_s_elements(mode='upstream')))
    assert np.all(np.array([1,2,3,4,5]) == np.array(line.get_s_elements(mode='downstream')))

    assert line.get_s_position(at_elements='e3') == 3.
    assert np.isscalar(line.get_s_position(at_elements='e3'))
    assert len(line.get_s_position(at_elements=['e3'])) == 1
    assert np.all(np.array([4,2]) == np.array(line.get_s_position(at_elements=['e4', 'e2'])))

    line.insert_element(element=xt.Cavity(), name="cav", at_s=3.3)
    assert line.get_length() == line.get_s_elements(mode='downstream')[-1] == 5
    assert line.get_s_position('cav') == 3.3
    assert len(line.elements) == 7

    line = line0.copy()
    line.insert_element(element=xt.Drift(length=0.2), at_s=0.11, name='inserted_drift')
    assert line.get_s_position('inserted_drift') == 0.11
    assert len(line.elements) == 7
    assert np.all([nn==nnref for nn, nnref in list(zip(line.element_names,
                ['e0..0', 'inserted_drift', 'e0..2', 'e1', 'e2', 'e3', 'e4']))])
    assert line.get_length() == line.get_s_elements(mode='downstream')[-1] == 5

    line = line0.copy()
    line.insert_element(element=xt.Drift(length=0.2), at_s=0.95, name='inserted_drift')
    assert line.get_s_position('inserted_drift') == 0.95
    assert len(line.elements) == 6
    assert np.all([nn==nnref for nn, nnref in list(zip(line.element_names,
                ['e0..0', 'inserted_drift', 'e1..1', 'e2', 'e3', 'e4']))])
    assert line.get_length() == line.get_s_elements(mode='downstream')[-1] == 5

    line = line0.copy()
    line.insert_element(element=xt.Drift(length=0.2), at_s=1.0, name='inserted_drift')
    assert line.get_s_position('inserted_drift') == 1.
    assert len(line.elements) == 6
    assert np.all([nn==nnref for nn, nnref in list(zip(line.element_names,
                ['e0', 'inserted_drift', 'e1..1', 'e2', 'e3', 'e4']))])
    assert line.get_length() == line.get_s_elements(mode='downstream')[-1] == 5

    line = line0.copy()
    line.insert_element(element=xt.Drift(length=0.2), at_s=0.8, name='inserted_drift')
    assert line.get_s_position('inserted_drift') == 0.8
    assert len(line.elements) == 6
    assert np.all([nn==nnref for nn, nnref in list(zip(line.element_names,
                ['e0..0', 'inserted_drift', 'e1', 'e2', 'e3', 'e4']))])
    assert line.get_length() == line.get_s_elements(mode='downstream')[-1] == 5

    line = line0.copy()
    line.insert_element(element=xt.LimitEllipse(a=1, b=1), at_s=2.1, name='aper')
    assert line.get_s_position('aper') == 2.1
    assert line.get_length() == line.get_s_elements(mode='downstream')[-1] == 5
    assert np.all([nn==nnref for nn, nnref in list(zip(line.element_names,
                ['e0', 'e1', 'e2..0', 'aper', 'e2..1', 'e3', 'e4']))])
    line.insert_element(element=xt.Drift(length=0.8), at_s=1.9, name="newdrift")
    assert line.get_s_position('newdrift') == 1.9
    assert np.all([nn==nnref for nn, nnref in list(zip(line.element_names,
                ['e0', 'e1..0', 'newdrift', 'e2..1..1', 'e3', 'e4']))])

    # Check preservation of markers
    elements = []
    enames = []

    for ii in range(5):
        elements.append(xt.Drift(length=1))
        enames.append(f'd{ii}')
        elements.append(xt.Drift(length=0))
        enames.append(f'm{ii}')

    line = xt.Line(elements=elements, element_names=enames)
    line.insert_element(element=xt.Drift(length=1.), at_s=1.0, name='inserted_drift')
    assert line.get_s_position('inserted_drift') == 1.
    assert len(line.elements) == 10
    assert np.all([nn==nnref for nn, nnref in list(zip(line.element_names,
        ['d0', 'm0', 'inserted_drift', 'm1', 'd2', 'm2', 'd3', 'm3', 'd4', 'm4']))])
    assert line.get_length() == line.get_s_elements(mode='downstream')[-1] == 5

    line.insert_element(element=xt.Cavity(), at_s=3.0, name='cav0')
    line.insert_element(element=xt.Cavity(), at_s=3.0, name='cav1')
    assert len(line.elements) == 12
    assert np.all([nn==nnref for nn, nnref in list(zip(line.element_names,
        ['d0', 'm0', 'inserted_drift', 'm1', 'd2', 'cav1', 'cav0', 'm2', 'd3',
        'm3', 'd4', 'm4']))])
    assert line.get_length() == line.get_s_elements(mode='downstream')[-1] == 5
    assert line.get_s_position('cav0') == 3.
    assert line.get_s_position('cav1') == 3.

    line = xt.Line(elements=elements, element_names=enames)
    line.insert_element(element=xt.Drift(length=0.2), at_s=0.95, name='inserted_drift')
    assert line.get_s_position('inserted_drift') == 0.95
    assert len(line.elements) == 10
    assert np.all([nn==nnref for nn, nnref in list(zip(line.element_names,
                ['d0..0', 'inserted_drift', 'd1..1', 'm1', 'd2', 'm2', 'd3',
                'm3', 'd4', 'm4']))])
    assert line.get_length() == line.get_s_elements(mode='downstream')[-1] == 5


def test_insert_omp():
    ctx = xo.ContextCpu(omp_num_threads='auto')
    buffer = ctx.new_buffer()

    drift = xt.Drift(length=2, _buffer=buffer)
    multipole = xt.Multipole(knl=[1], _buffer=buffer)

    line = xt.Line(elements=[drift], element_names=['dr'])
    line.insert_element(element=multipole, at_s=1, name='mp')
    line.build_tracker()

    assert line._buffer is line['dr..0']._buffer
    assert line['dr..0']._buffer is line['mp']._buffer
    assert line['mp']._buffer is line['dr..1']._buffer
    assert line._context.omp_num_threads == 'auto'


def test_to_pandas():
    line = xt.Line(elements=[
        xt.Drift(length=1), xt.Cavity(), xt.Drift(length=1)])

    df = line.to_pandas()

    assert tuple(df.columns) == (
        's', 'element_type', 'name', 'isthick', 'isreplica', 'parent_name',
       'iscollective', 'element', 's_start', 's_center', 's_end')
    assert len(df) == 4

def test_check_aperture():

    class ThickElement:
        length = 2.
        isthick = True

    line = xt.Line(
        elements={
            'dum': xt.Drift(length=0),
            'dr1': xt.Drift(length=1),
            'm1_ap': xt.LimitEllipse(a=1e-2, b=1e-2),
            'm1': xt.Multipole(knl=[1]),
            'dr2': xt.Drift(length=1),
            'm2': xt.Multipole(knl=[1]),
            'dr3': xt.Drift(length=1),
            'th1_ap_front': xt.LimitEllipse(a=1e-2, b=1e-2),
            'th1': ThickElement(),
            'th1_ap_back': xt.LimitEllipse(a=1e-2, b=1e-2),
            'dr4': xt.Drift(length=1),
            'th2': ThickElement(),
            'th2_ap_back': xt.LimitEllipse(a=1e-2, b=1e-2),
            'dr5': xt.Drift(length=1),
            'th3_ap_front': xt.LimitEllipse(a=1e-2, b=1e-2),
            'th3': ThickElement(),
            'dr6': xt.Drift(length=1),
            'th4_ap_entry': xt.LimitEllipse(a=1e-2, b=1e-2),
            'th4': ThickElement(),
            'th4_ap_exit': xt.LimitEllipse(a=1e-2, b=1e-2),
        },
        element_names=['dr1', 'm1_ap', 'dum', 'm1', 'dr2', 'm2', 'dr3',
                       'th1_ap_front', 'dum', 'th1', 'dum', 'th1_ap_back',
                       'dr4', 'th2', 'th2_ap_back',
                       'dr5', 'th3_ap_front', 'th3', 'dr6',
                       'th4_ap_entry', 'th4', 'th4_ap_exit'])

    df = line.check_aperture()

    expected_miss_upstream = [nn in ('m2', 'th2') for nn in df['name'].values]
    expected_miss_downstream = [nn in ('m1', 'm2', 'th3') for nn in df['name'].values]
    expected_problem_flag = np.array(expected_miss_upstream) | (df.isthick &
                                        np.array(expected_miss_downstream))

    assert np.all(df['misses_aperture_upstream'].values == expected_miss_upstream)
    assert np.all(df['misses_aperture_downstream'].values == expected_miss_downstream)
    assert np.all(df['has_aperture_problem'].values == expected_problem_flag)

def test_to_dict():
    line = xt.Line(
        elements={
            'm': xt.Multipole(knl=[1, 2]),
            'd': xt.Drift(length=1),
        },
        element_names=['m', 'd', 'm', 'd']
    )
    result = line.to_dict()

    assert len(result['elements']) == 2
    assert result['element_names'] == ['m', 'd', 'm', 'd']

    assert result['elements']['m']['__class__'] == 'Multipole'
    assert (result['elements']['m']['knl'] == [1, 2]).all()

    assert result['elements']['d']['__class__'] == 'Drift'
    assert result['elements']['d']['length'] == 1

    assert result['metadata'] == line.metadata

def test_from_dict_legacy():
    test_dict = {
        'elements': [
            {'__class__': 'Multipole', 'knl': [1, 2]},
            {'__class__': 'Drift', 'length': 1},
        ],
        'element_names': ['mn1', 'd1'],
    }
    result = xt.Line.from_dict(test_dict)

    assert len(result.elements) == 2

    assert isinstance(result.elements[0], xt.Multipole)
    assert (result.elements[0].knl == [1, 2]).all()

    assert isinstance(result.elements[1], xt.Drift)
    assert result.elements[1].length == 1

    assert result.element_names == ['mn1', 'd1']

@pytest.mark.parametrize('under_test', ['copy', 'from_to_dict'])
def test_copy_to_dict_from_dict_config(under_test):

    # Step 1: Create a line object
    line = xt.Line()

    # Step 2: Build the tracker
    line.build_tracker()
    line.metadata['hello'] = 'world'

    # Step 3: Configure radiation model
    line.configure_radiation(model='mean')

    # Step 5: Serialize and deserialize the line
    if under_test == 'copy':
        l2 = line.copy()
    elif under_test == 'from_to_dict':
        line_dict = line.to_dict()
        l2 = xt.Line.from_dict(line_dict)
    else:
        raise ValueError(f'Unknown test {under_test}')

    assert np.all(
        np.sort(list(line.config.keys())) == np.sort(list(l2.config.keys())))

    for key in line.config.keys():
        assert line.config[key] == l2.config[key]

    assert line.metadata['hello'] == l2.metadata['hello']


def test_from_dict_current():
    test_dict = {
        'elements': {
            'mn': {
                '__class__': 'Multipole',
                'knl': [1, 2],
            },
            'ms': {
                '__class__': 'Multipole',
                'ksl': [3],
            },
            'd': {
                '__class__': 'Drift',
                'length': 4,
            },
        },
        'element_names': ['mn', 'd', 'ms', 'd'],
        'metadata' : {
            'config_knobs_and_tuning': {
                'knob_settings': {
                    'on_x1': 135.0,
                },
            },
        },
    }
    line = xt.Line.from_dict(test_dict)

    assert line.element_names == ['mn', 'd', 'ms', 'd']
    mn, d1, ms, d2 = line.elements

    assert isinstance(mn, xt.Multipole)
    assert (mn.knl == [1, 2]).all()

    assert isinstance(ms, xt.Multipole)
    assert (ms.ksl == [3]).all()

    assert isinstance(d1, xt.Drift)
    assert d1.length == 4

    assert d2 is d1

    assert line.metadata == test_dict['metadata']


def test_from_sequence():
    # direct element definition
    # -------------------------
    line = Line.from_sequence([
        Node(3, Multipole(knl=[1])),
        Node(7, Multipole(), name='quad1'),
        Node(1, Multipole(), name='bend1', from_='quad1'),
        ], 10)
    assert line.get_length() == 10
    assert len(line.elements) == 7
    for i, l in enumerate([3, 0, 4, 0, 1, 0, 2]):
        cls = xt.Multipole if i%2 else xt.Drift
        assert isinstance(line.elements[i], cls)
        xo.assert_allclose(line.elements[i].length, l)

    # using pre-defined elements by name
    # ----------------------------------
    elements = {
        'quad': Multipole(length=0.3, knl=[0, +0.50]),
        'bend': Multipole(length=0.5, knl=[np.pi / 12], hxl=[np.pi / 12]),
    }
    line = Line.from_sequence(length=10, nodes=[
        Node(1, 'quad'),
        Node(1, 'quad', name='quad3', from_=3),
        Node(2, 'bend', from_='quad3', name='bend2'),
    ], elements=elements, auto_reorder=True)
    assert line.get_length() == 10
    assert len(line.elements) == 7
    for i, l in enumerate([1, 0.3, 3, 0.3, 2, 0.5, 4]):
        cls = xt.Multipole if i%2 else xt.Drift
        assert isinstance(line.elements[i], cls)
        xo.assert_allclose(line.elements[i].length, l)
    assert line.elements[1] == line.elements[3]
    assert line.element_names[1] == 'quad'
    assert line.element_names[3] == 'quad3'
    assert line.element_names[5] == 'bend2'
    assert line.elements[5] == elements['bend']

    # using nested sequences
    # ----------------------
    sequences = {
        'arc': [Node(1, 'quad'), Node(4, 'bend', from_='quad')],
    }
    sext = Multipole(knl=[0, 0, 0.1])
    line = Line.from_sequence([
        Node(0, 'arc', name='section_1'),
        Node(10, 'arc', name='section_2'),
        Node(3, sext, from_='section_2', name='sext'),
        Node(3, [Node(1, 'quad')], name='section_3', from_='sext'),
    ], length=20, elements=elements, sequences=sequences, auto_reorder=True, naming_scheme='{}_{}')

    assert line.get_length() == 20
    assert len(line.elements) == 18

    assert line.get_s_position()[line.element_names.index('section_1')] == 0
    assert isinstance(line.elements[line.element_names.index('section_1')], xt.Marker)
    assert line.get_s_position()[line.element_names.index('section_1_quad')] == 1
    assert line.elements[line.element_names.index('section_1_quad')] == elements['quad']
    assert line.get_s_position()[line.element_names.index('section_1_bend')] == 5
    assert line.elements[line.element_names.index('section_1_bend')] == elements['bend']

    assert line.get_s_position()[line.element_names.index('section_2')] == 10
    assert isinstance(line.elements[line.element_names.index('section_2')], xt.Marker)
    assert line.get_s_position()[line.element_names.index('section_2_quad')] == 11
    assert line.elements[line.element_names.index('section_2_quad')] == elements['quad']
    assert line.get_s_position()[line.element_names.index('section_2_bend')] == 15
    assert line.elements[line.element_names.index('section_2_bend')] == elements['bend']

    assert line.get_s_position()[line.element_names.index('sext')] == 13
    assert line.elements[line.element_names.index('sext')] == sext

    assert line.get_s_position()[line.element_names.index('section_3')] == 16
    assert isinstance(line.elements[line.element_names.index('section_3')], xt.Marker)
    assert line.get_s_position()[line.element_names.index('section_3_quad')] == 17
    assert line.elements[line.element_names.index('section_3_quad')] == elements['quad']

    # test negative drift
    # -------------------
    Line.from_sequence([Node(3, Multipole()), Node(2, Multipole())], 10, auto_reorder=True)

    with pytest.raises(ValueError):
        Line.from_sequence([Node(3, Multipole()), Node(2, Multipole())], 10)

    with pytest.raises(ValueError):
        Line.from_sequence([Node(1, Multipole()), Node(4, Multipole())], 2)


@pytest.mark.parametrize('refer', ['entry', 'centre', 'exit'])
def test_from_sequence_with_thick(refer):
    sequence = [
        xt.Node(1.2, xt.Drift(length=1), name='my_drift'),
        xt.Node(3, xt.Bend(length=1, k0=0.2), name='my_bend'),
    ]
    line = xt.Line.from_sequence(sequence, 5, refer=refer)  # noqa

    assert len(line) == 5
    assert line.get_length() == 5.0

    assert line.element_names[1] == 'my_drift'
    assert line.element_names[3] == 'my_bend'

    offset = 0
    if refer == 'centre':
        offset = -0.5
    elif refer == 'exit':
        offset = -1

    xo.assert_allclose(
        line.get_s_position(line.element_names),
        [
            0,             # drift
            1.2 + offset,  # my_drift
            2.2 + offset,  # drift
            3 + offset,    # my_bend
            4 + offset,    # drift
        ],
        atol=1e-15,
    )


def test_from_sequence_with_thick_fails():
    sequence = [
        xt.Node(1.2, xt.Drift(length=3), name='my_drift'),
        xt.Node(3, xt.Bend(length=3, k0=0.2), name='my_bend'),
    ]
    with pytest.raises(ValueError):
        _ = xt.Line.from_sequence(sequence, 5)


@for_all_test_contexts
def test_optimize_multipoles(test_context):
    elements = {
        'q1': xt.Multipole(knl=[0, 1], length=0, _context=test_context),
        'q2': xt.Multipole(knl=[1, 2], length=0, _context=test_context),
        'q3': xt.Multipole(knl=[0, 1], length=1, _context=test_context),
        'q4': xt.Multipole(knl=[0, 1], ksl=[1, 2], length=0, _context=test_context),
        'd1': xt.Multipole(knl=[1], hxl=0.0, length=2, _context=test_context),
        'd2': xt.Multipole(knl=[1], hxl=0.1, length=0, _context=test_context),
        'd4': xt.Multipole(knl=[1], ksl=[3], length=2, _context=test_context),
    }

    test_line = xt.Line(
        elements=elements,
        element_names=elements.keys(),
    )

    test_line.use_simple_bends()
    test_line.use_simple_quadrupoles()

    for nn in test_line.element_names:
        if nn in ('d1', 'd2'):
            assert type(test_line.element_dict[nn]) is xt.SimpleThinBend
        elif nn == 'q1' or nn == 'q3':
            assert type(test_line.element_dict[nn]) is xt.SimpleThinQuadrupole
        else:
            assert type(test_line.element_dict[nn]) is xt.Multipole

def test_from_json_to_json(tmp_path):

    line = xt.Line(
        elements={
            'm': xt.Multipole(knl=[1, 2]),
            'd': xt.Drift(length=1),
        },
        element_names=['m', 'd', 'm', 'd']
    )

    example_metadata = {
        'qx': {'lhcb1': 62.31, 'lhcb2': 62.31},
        'delta_cmr': 0.0,
    }
    line.metadata = example_metadata

    def asserts():
        assert len(result.element_dict.keys()) == 2
        assert result.element_names == ['m', 'd', 'm', 'd']

        assert isinstance(result['m'], xt.Multipole)
        assert (result['m'].knl == [1, 2]).all()

        assert isinstance(result['d'], xt.Drift)
        assert result['d'].length == 1

        assert result.metadata == example_metadata
        result.metadata['qx']['lhcb1'] = result.metadata['qx']['lhcb1'] + 1
        assert result.metadata != example_metadata
        result.metadata['qx']['lhcb1'] = result.metadata['qx']['lhcb1'] - 1

    line.to_json(tmp_path / 'test.json')
    result = xt.load(tmp_path / 'test.json')

    asserts()

    with open(tmp_path / 'test2.json', 'w') as f:
        line.to_json(f)

    with open(tmp_path / 'test2.json', 'r') as f:
        result = xt.Line.from_json(f)

    asserts()

    with open(tmp_path / 'test2.json', 'w') as f:
        line.to_json(f,indent=None)

    with open(tmp_path / 'test2.json', 'r') as f:
        result = xt.Line.from_json(f)

    asserts()

    with open(tmp_path / 'test2.json.gz', 'w') as f:
        line.to_json(f,indent=2)

    with open(tmp_path / 'test2.json.gz', 'r') as f:
        result = xt.Line.from_json(f)

    asserts()


@for_all_test_contexts
def test_config_propagation(test_context):
    line = xt.Line(elements=10*[xt.Drift(length=1)])
    line.config.TEST1 = True
    line.config.TEST2 = 33.3
    line.matrix_stability_tol = 55.5
    dct = line.to_dict()

    assert 'config' in dct
    assert '_extra_config' in dct

    line2 = xt.Line.from_dict(dct)
    assert line2.config.TEST1 == True
    assert line2.config.TEST2 == 33.3
    assert line2.matrix_stability_tol == 55.5

    line2.build_tracker(_context=test_context)
    assert line2.tracker.matrix_stability_tol == 55.5

    # Check that they are copies
    line2.config.TEST1 = 23.3
    assert line2.config.TEST1 == 23.3
    assert line.config.TEST1 == True
    line2.matrix_stability_tol = 77.7
    assert line2.matrix_stability_tol == 77.7
    assert line.matrix_stability_tol == 55.5

    line3 = line.copy()
    assert line3.config.TEST1 == True
    assert line3.config.TEST2 == 33.3
    assert line3.matrix_stability_tol == 55.5

    # Check that they are copies
    line3.config.TEST1 = 23.3
    assert line3.config.TEST1 == 23.3
    assert line.config.TEST1 == True
    line3.matrix_stability_tol = 77.7
    assert line3.matrix_stability_tol == 77.7
    assert line.matrix_stability_tol == 55.5


def test_pickle():

    # Load the line
    line = xt.load(test_data_folder /
            'hllhc15_noerrors_nobb/line_w_knobs_and_particle.json')
    line.particle_ref = xp.Particles(p0c=7e12, mass=xp.PROTON_MASS_EV)
    line.build_tracker()

    lnss = pickle.dumps(line)
    ln = pickle.loads(lnss)

    # Check that expressions work on old and new line
    line.vars['on_x1'] = 234
    ln.vars['on_x1'] = 123
    xo.assert_allclose(line.twiss(method='4d')['px', 'ip1'], 234e-6, atol=1e-9, rtol=0)
    xo.assert_allclose(ln.twiss(method='4d')['px', 'ip1'], 123e-6, atol=1e-9, rtol=0)

    ln.vars['on_x1'] = 321
    xo.assert_allclose(line.twiss(method='4d')['px', 'ip1'], 234e-6, atol=1e-9, rtol=0)
    xo.assert_allclose(ln.twiss(method='4d')['px', 'ip1'], 321e-6, atol=1e-9, rtol=0)

    line.vars['on_x1'] = 213
    xo.assert_allclose(line.twiss(method='4d')['px', 'ip1'], 213e-6, atol=1e-9, rtol=0)
    xo.assert_allclose(ln.twiss(method='4d')['px', 'ip1'], 321e-6, atol=1e-9, rtol=0)

    line.discard_tracker()

    collider = xt.Environment(lines={'lhcb1': line})
    collider.build_trackers()

    colliderss = pickle.dumps(collider)

    coll = pickle.loads(colliderss)

    collider.vars['on_x1'] = 234
    coll.vars['on_x1'] = 123
    xo.assert_allclose(collider['lhcb1'].twiss(method='4d')['px', 'ip1'], 234e-6, atol=1e-9, rtol=0)
    xo.assert_allclose(coll['lhcb1'].twiss(method='4d')['px', 'ip1'], 123e-6, atol=1e-9, rtol=0)

    coll.vars['on_x1'] = 321
    xo.assert_allclose(collider['lhcb1'].twiss(method='4d')['px', 'ip1'], 234e-6, atol=1e-9, rtol=0)
    xo.assert_allclose(coll['lhcb1'].twiss(method='4d')['px', 'ip1'], 321e-6, atol=1e-9, rtol=0)

    collider.vars['on_x1'] = 213
    xo.assert_allclose(collider['lhcb1'].twiss(method='4d')['px', 'ip1'], 213e-6, atol=1e-9, rtol=0)
    xo.assert_allclose(coll['lhcb1'].twiss(method='4d')['px', 'ip1'], 321e-6, atol=1e-9, rtol=0)


def test_line_attr():
    line = xt.Line(
        elements=[
            xt.Drift(length=1),
            xt.Multipole(knl=[2, 3, 4], hxl=8),
            xt.Bend(k0=5, h=0.5, length=6, knl=[7, 8, 9]),
            xt.Drift(length=10),
            xt.Quadrupole(k1=11, length=12),
        ]
    )

    line.build_tracker()

    assert np.all(line.attr['length'] == [1, 0, 6, 10, 12])
    assert np.all(line.attr['k0l'] == [0, 2, 5 * 6 + 7, 0, 0])
    assert np.all(line.attr['k1l'] == [0, 3, 8, 0, 11 * 12])
    assert np.all(line.attr['angle_rad'] == [0, 8, 0.5 * 6, 0, 0])

@for_all_test_contexts
def test_insert_thin_elements_at_s_basic(test_context):

    l1 = xt.Line(elements=5*[xt.Drift(length=1)])
    l1.build_tracker(_context=test_context) # Move all elements to selected context

    l1.discard_tracker()
    l1._insert_thin_elements_at_s([
        (0, [(f'm0_at_a', xt.Marker(_context=test_context)), (f'm1_at_a', xt.Marker(_context=test_context))]),
        (5, [(f'm0_at_b', xt.Marker(_context=test_context)), (f'm1_at_b', xt.Marker(_context=test_context))]),
    ])

    t1 = l1.get_table()
    assert t1.name[0] == 'm0_at_a'
    assert t1.name[1] == 'm1_at_a'
    assert t1.name[-1] == '_end_point'
    assert t1.name[-2] == 'm1_at_b'
    assert t1.name[-3] == 'm0_at_b'

    assert t1.s[0] == 0
    assert t1.s[1] == 0
    assert t1.s[-1] == 5.
    assert t1.s[-2] == 5.
    assert t1.s[-3] == 5.

@for_all_test_contexts
def test_insert_thin_elements_at_s_lhc(test_context):

    line = xt.load(test_data_folder /
                    'hllhc15_thick/lhc_thick_with_knobs.json')
    line.twiss_default['method'] = '4d'

    line.build_tracker(_context=test_context)

    Strategy = xt.Strategy
    Teapot = xt.Teapot
    slicing_strategies = [
        Strategy(slicing=Teapot(1)),  # Default catch-all as in MAD-X
        Strategy(slicing=None, element_type=xt.UniformSolenoid),
        Strategy(slicing=Teapot(4), element_type=xt.Bend),
        Strategy(slicing=Teapot(20), element_type=xt.Quadrupole),
        Strategy(slicing=Teapot(2), name=r'^mb\..*'),
        Strategy(slicing=Teapot(5), name=r'^mq\..*'),
        Strategy(slicing=Teapot(2), name=r'^mqt.*'),
        Strategy(slicing=Teapot(60), name=r'^mqx.*'),
    ]

    line.discard_tracker()
    line.slice_thick_elements(slicing_strategies=slicing_strategies)

    tw0 = line.twiss()
    line.discard_tracker()

    e0 = 'mq.28r3.b1_entry'
    e1 = 'mq.29r3.b1_exit'

    s0 = line.get_s_position(e0)
    s1 = line.get_s_position(e1)
    s2 = line.get_length()

    elements_to_insert = [
        # s .    # elements to insert (name, element)
        (s0,     [(f'm0_at_a', xt.Marker(_context=test_context)), (f'm1_at_a', xt.Marker(_context=test_context)), (f'm2_at_a', xt.Marker(_context=test_context))]),
        (s0+10., [(f'm0_at_b', xt.Marker(_context=test_context)), (f'm1_at_b', xt.Marker(_context=test_context)), (f'm2_at_b', xt.Marker(_context=test_context))]),
        (s1,     [(f'm0_at_c', xt.Marker(_context=test_context)), (f'm1_at_c', xt.Marker(_context=test_context)), (f'm2_at_c', xt.Marker(_context=test_context))]),
        (s2,     [(f'm0_at_d', xt.Marker(_context=test_context)), (f'm1_at_d', xt.Marker(_context=test_context)), (f'm2_at_d', xt.Marker(_context=test_context))]),
    ]

    line.discard_tracker()
    line._insert_thin_elements_at_s(elements_to_insert)
    line.build_tracker(_context=test_context)

    tt = line.get_table()

    # Check that there are no duplicated elements
    assert len(tt.name) == len(set(tt.name))

    xo.assert_allclose(tt['s', 'm0_at_a'], s0, rtol=0, atol=1e-6)
    xo.assert_allclose(tt['s', 'm1_at_a'], s0, rtol=0, atol=1e-6)
    xo.assert_allclose(tt['s', 'm2_at_a'], s0, rtol=0, atol=1e-6)

    xo.assert_allclose(tt['s', 'm0_at_b'], s0 + 10., rtol=0, atol=1e-6)
    xo.assert_allclose(tt['s', 'm1_at_b'], s0 + 10., rtol=0, atol=1e-6)
    xo.assert_allclose(tt['s', 'm2_at_b'], s0 + 10., rtol=0, atol=1e-6)

    xo.assert_allclose(tt['s', 'm0_at_c'], s1, rtol=0, atol=1e-6)
    xo.assert_allclose(tt['s', 'm1_at_c'], s1, rtol=0, atol=1e-6)
    xo.assert_allclose(tt['s', 'm2_at_c'], s1, rtol=0, atol=1e-6)

    xo.assert_allclose(tt['s', 'm0_at_d'], s2, rtol=0, atol=1e-6)
    xo.assert_allclose(tt['s', 'm1_at_d'], s2, rtol=0, atol=1e-6)
    xo.assert_allclose(tt['s', 'm2_at_d'], s2, rtol=0, atol=1e-6)

    assert np.all(tt.rows['mq.28r3.b1_entry<<3':'mq.28r3.b1_entry'].name
            == np.array(['m0_at_a', 'm1_at_a', 'm2_at_a', 'mq.28r3.b1_entry']))

    assert np.all(tt.rows['m0_at_b<<2':'m0_at_b>>4'].name
            == np.array(['mb.a29r3.b1..0', 'drift_mb.a29r3.b1..1..0',
                        'm0_at_b', 'm1_at_b', 'm2_at_b',
                        'drift_mb.a29r3.b1..1..1', 'mb.a29r3.b1..1']))

    assert np.all(tt.rows['mq.29r3.b1_exit<<3':'mq.29r3.b1_exit'].name
            == np.array(
                ['m1_at_c', 'm2_at_c', 'mq.29r3.b1..exit_map', 'mq.29r3.b1_exit']))

    assert np.all(tt.rows['m0_at_d':'m0_at_d>>4'].name
                == np.array(['m0_at_d', 'm1_at_d', 'm2_at_d',
                            'lhcb1ip7_p_', '_end_point']))

    xo.assert_allclose(line.get_length(), tw0.s[-1], atol=1e-6)

    tw1 = line.twiss()
    xo.assert_allclose(tw1.qx, tw0.qx, atol=1e-9, rtol=0)


def test_elements_intersecting_s():
    elements = {
        'e1': xt.Drift(length=1),  # at 0
        'e2': xt.Drift(length=2),  # at 1
        'm1': xt.Marker(),         # at 3
        'e3': xt.Drift(length=1),  # at 3
        'm2': xt.Marker(),         # at 4
        'm3': xt.Marker(),         # at 4
        'e4': xt.Drift(length=1),  # at 4
        'e5': xt.Drift(length=2),  # at 5
        'e6': xt.Drift(length=1),  # at 7
        'm4': xt.Marker(),         # at 8
    }

    line = xt.Line(
        elements=elements,
        element_names=list(elements.keys()),
    )

    cuts = [
        0.5,  # e1
        1,
        1.1, 1.7,  # e2
        4.5,  # e4
        7.8, 7.9,  # e6
        8,
        8,
    ]

    expected = {
        'e1': [0.5],
        'e2': [0.1, 0.7],
        'e4': [0.5],
        'e6': [0.8, 0.9],
    }
    result = line._elements_intersecting_s(cuts)
    for kk in expected.keys() | result.keys():
        xo.assert_allclose(expected[kk], result[kk], atol=1e-14)


def test_slicing_at_custom_s():
    elements = {
        'e1': xt.Drift(length=1),  # at 0
        'e2': xt.Drift(length=2),  # at 1
        'm1': xt.Marker(),         # at 3
        'e3': xt.Drift(length=1),  # at 3
        'm2': xt.Marker(),         # at 4
        'm3': xt.Marker(),         # at 4
        'e4': xt.Drift(length=1),  # at 4
        'e5': xt.Drift(length=2),  # at 5
        'e6': xt.Drift(length=1),  # at 7
        'm4': xt.Marker(),         # at 8
    }

    line = xt.Line(
        elements=elements,
        element_names=list(elements.keys()),
    )

    cuts = [
        0.5,  # e1
        1,
        1.1, 1.7,  # e2
        4.5,  # e4
        7.8, 7.9,  # e6
        8,
        8,
    ]

    line.cut_at_s(cuts)

    tab = line.get_table()
    xo.assert_allclose(tab.rows[r'e1\.\.\d*'].s, [0, 0.5], atol=1e-16)
    xo.assert_allclose(tab.rows[r'e2\.\.\d*'].s, [1, 1.1, 1.7], atol=1e-16)
    xo.assert_allclose(tab.rows[r'e3'].s, [3], atol=1e-16)
    xo.assert_allclose(tab.rows[r'e4\.\.\d*'].s, [4, 4.5], atol=1e-16)
    xo.assert_allclose(tab.rows[r'e5'].s, [5], atol=1e-16)
    xo.assert_allclose(tab.rows[r'e6\.\.\d*'].s, [7, 7.8, 7.9], atol=1e-16)

def test_insert_thick_element_reuse_marker_name():

    assert_allclose = xo.assert_allclose

    elements = {
        'd1': xt.Drift(length=1),
        'm1': xt.Marker(),
        'd2': xt.Drift(length=1),
    }

    line=xt.Line(elements=elements,
                element_names=list(elements.keys()))

    # Note that the name is reused
    line.insert_element(element=xt.Bend(length=1.), name='m1', at_s=0.5)

    tt = line.get_table()

    assert np.all(tt.name == ['d1..0', 'm1', 'd2..1', '_end_point'])
    assert np.all(tt.parent_name == ['d1', None, 'd2', None])
    assert_allclose(tt.s, [0. , 0.5, 1.5, 2. ], rtol=0, atol=1e-14)

def test_multiple_thick_elements():
    line = xt.Line(
        elements=[xt.Drift(length=1.0) for i in range(10)]
    )

    s_insert = np.array([2.5, 5.5, 7])
    l_insert = np.array([1.0, 1.0, 1.0])
    ele_insert = [xt.Sextupole(length=l) for l in l_insert]

    line._insert_thick_elements_at_s(
        element_names=[f'insertion_{i}' for i in range(len(s_insert))],
        elements=ele_insert,
        at_s=s_insert
    )

    tt = line.get_table()

    assert np.all(tt.name == ['e0', 'e1', 'e2..0', 'insertion_0', 'e3..1', 'e4', 'e5..0',
        'insertion_1', 'e6..1', 'insertion_2', 'e8', 'e9', '_end_point'])
    xo.assert_allclose(tt.s, [ 0. ,  1. ,  2. ,  2.5,  3.5,  4. ,  5. ,
                                    5.5,  6.5,  7. ,  8. , 9. , 10. ])

    assert np.all(tt.element_type == ['Drift', 'Drift', 'DriftSlice', 'Sextupole', 'DriftSlice', 'Drift',
        'DriftSlice', 'Sextupole', 'DriftSlice', 'Sextupole', 'Drift',
        'Drift', ''])

@for_all_test_contexts
def test_get_strengths(test_context):
    collider = xt.load(
        test_data_folder / 'hllhc15_thick/hllhc15_collider_thick.json')
    collider.build_trackers(_context=test_context)

    collider.lhcb1.twiss_default['method'] = '4d'
    collider.lhcb2.twiss_default['method'] = '4d'
    collider.lhcb2.twiss_default['reverse'] = True

    line = collider.lhcb2 # <- use lhcb2 to test the reverse option

    import xobjects as xo
    str_table_rev = line.get_strengths() # Takes reverse from twiss_default
    xo.assert_allclose(line['mbw.a6l3.b2'].k0,
            -str_table_rev['k0l', 'mbw.a6l3.b2'] / str_table_rev['length', 'mbw.a6l3.b2'],
            rtol=0, atol=1e-14)
    xo.assert_allclose(line['mbw.a6l3.b2'].h,
            -str_table_rev['angle_rad', 'mbw.a6l3.b2'] / str_table_rev['length', 'mbw.a6l3.b2'],
            rtol=0, atol=1e-14)

    str_table = line.get_strengths(reverse=False) # Takes reverse from twiss_default
    xo.assert_allclose(line['mbw.a6l3.b2'].k0,
            str_table['k0l', 'mbw.a6l3.b2'] / str_table['length', 'mbw.a6l3.b2'],
            rtol=0, atol=1e-14)
    xo.assert_allclose(line['mbw.a6l3.b2'].h,
            str_table['angle_rad', 'mbw.a6l3.b2'] / str_table['length', 'mbw.a6l3.b2'],
            rtol=0, atol=1e-14)



def test_insert_repeated_names():

    line = xt.Line(
        elements=([xt.Drift(length=0)] # Start line marker
                    + [xt.Drift(length=1) for _ in range(5)]
                    + [xt.Drift(length=0)] # End line marker
            ),
        element_names=['d']*7
        )
    line.insert_element("m1",xt.Marker(),at="d::3")
    assert line.element_names[3]=="m1"
    line.insert_element("m2",xt.Marker(),at="d")
    assert line.element_names[0]=="m2"

def test_line_table_unique_names():
    line = xt.Line(
        elements = {"obm": xt.Bend(length=0.5)},
        element_names= ["obm","obm"]
    )
    table = line.get_table()
    names, counts = np.unique(table.name, return_counts=True, equal_nan=False)
    assert np.all(counts == 1), "Not all elements are unique"
    for name, env_name in zip(table.name, table.env_name):
        if name == '_end_point': continue
        assert line[name] == line[env_name]


def test_extend_knl_ksl():

    classes_to_check = ['Bend', 'Quadrupole', 'Sextupole', 'Octupole', 'UniformSolenoid',
                        'VariableSolenoid', 'Multipole']

    for cc in classes_to_check:

        nn1 = 'test1_'+cc.lower()
        nn2 = 'test2_'+cc.lower()
        env = xt.Environment()
        env.new(nn1, cc, length=10, knl=[
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], ksl=[3, 2, 1])
        env.new(nn2, cc, length=10, ksl=[
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], knl=[3, 2, 1], order=11)

        assert env[nn1].__class__.__name__ == cc
        assert env[nn1].order == 11
        assert len(env[nn1].knl) == 12
        assert len(env[nn1].ksl) == 12
        xo.assert_allclose(env[nn1].knl, [1, 2, 3, 4, 5,
                           6, 7, 8, 9, 10, 11, 12], rtol=0, atol=1e-15)
        xo.assert_allclose(env[nn1].ksl, [3, 2, 1, 0, 0,
                           0, 0, 0, 0, 0, 0, 0], rtol=0, atol=1e-15)
        xo.assert_allclose(env[nn1].inv_factorial_order,
                           1/math.factorial(11), rtol=0, atol=1e-15)

        assert env[nn2].__class__.__name__ == cc
        assert env[nn2].order == 11
        assert len(env[nn2].ksl) == 12
        assert len(env[nn2].knl) == 12
        xo.assert_allclose(env[nn2].ksl, [1, 2, 3, 4, 5,
                           6, 7, 8, 9, 10, 11, 12], rtol=0, atol=1e-15)
        xo.assert_allclose(env[nn2].knl, [3, 2, 1, 0, 0,
                           0, 0, 0, 0, 0, 0, 0], rtol=0, atol=1e-15)
        xo.assert_allclose(env[nn2].inv_factorial_order,
                           1/math.factorial(11), rtol=0, atol=1e-15)

    env.vars.default_to_zero = True
    line = env.new_line(components=[
        env.new('b1', xt.Bend, length=1, knl=[
                'a', 'b', 'c'], ksl=['d', 'e', 'f']),
        env.new('q1', xt.Quadrupole, length=1, knl=[
                'a', 'b', 'c'], ksl=['d', 'e', 'f']),
        env.new('s1', xt.Sextupole, length=1, knl=[
                'a', 'b', 'c'], ksl=['d', 'e', 'f']),
        env.new('o1', xt.Octupole, length=1, knl=[
                'a', 'b', 'c'], ksl=['d', 'e', 'f']),
        env.new('u1', xt.UniformSolenoid, length=1, knl=[
                'a', 'b', 'c'], ksl=['d', 'e', 'f']),
        env.new('v1', xt.VariableSolenoid, length=1, knl=[
                'a', 'b', 'c'], ksl=['d', 'e', 'f']),
        env.new('m1', xt.Multipole, length=1, knl=[
                'a', 'b', 'c'], ksl=['d', 'e', 'f']),
    ])

    env['a'] = 3.
    env['b'] = 2.
    env['c'] = 1.
    env['d'] = 4.
    env['e'] = 5.
    env['f'] = 6.

    element_names = ['b1', 'q1']
    order = 10

    line.extend_knl_ksl(order=order, element_names=element_names)

    assert line['b1'].order == order
    assert line['q1'].order == order
    assert line['s1'].order == 5
    assert line['o1'].order == 5
    assert line['u1'].order == 5
    assert line['m1'].order == 2

    xo.assert_allclose(line['b1'].inv_factorial_order,
                       1/math.factorial(order), rtol=0, atol=1e-15)
    xo.assert_allclose(line['q1'].inv_factorial_order,
                       1/math.factorial(order), rtol=0, atol=1e-15)
    xo.assert_allclose(line['s1'].inv_factorial_order,
                       1/math.factorial(5), rtol=0, atol=1e-15)
    xo.assert_allclose(line['o1'].inv_factorial_order,
                       1/math.factorial(5), rtol=0, atol=1e-15)
    xo.assert_allclose(line['u1'].inv_factorial_order,
                       1/math.factorial(5), rtol=0, atol=1e-15)
    xo.assert_allclose(line['v1'].inv_factorial_order,
                       1/math.factorial(5), rtol=0, atol=1e-15)
    xo.assert_allclose(line['m1'].inv_factorial_order,
                       1/math.factorial(2), rtol=0, atol=1e-15)

    xo.assert_allclose(line['b1'].knl, [3., 2., 1., 0.,
                       0., 0., 0., 0., 0., 0., 0.], rtol=0, atol=1e-15)
    xo.assert_allclose(line['b1'].ksl, [4., 5., 6., 0.,
                       0., 0., 0., 0., 0., 0., 0.], rtol=0, atol=1e-15)
    xo.assert_allclose(line['q1'].knl, [3., 2., 1., 0.,
                       0., 0., 0., 0., 0., 0., 0.], rtol=0, atol=1e-15)
    xo.assert_allclose(line['q1'].ksl, [4., 5., 6., 0.,
                       0., 0., 0., 0., 0., 0., 0.], rtol=0, atol=1e-15)
    xo.assert_allclose(line['s1'].knl, [3., 2., 1.,
                       0., 0., 0.], rtol=0, atol=1e-15)
    xo.assert_allclose(line['s1'].ksl, [4., 5., 6.,
                       0., 0., 0.], rtol=0, atol=1e-15)
    xo.assert_allclose(line['o1'].knl, [3., 2., 1.,
                       0., 0., 0.], rtol=0, atol=1e-15)
    xo.assert_allclose(line['o1'].ksl, [4., 5., 6.,
                       0., 0., 0.], rtol=0, atol=1e-15)
    xo.assert_allclose(line['u1'].knl, [3., 2., 1.,
                       0., 0., 0.], rtol=0, atol=1e-15)
    xo.assert_allclose(line['u1'].ksl, [4., 5., 6.,
                       0., 0., 0.], rtol=0, atol=1e-15)
    xo.assert_allclose(line['v1'].knl, [3., 2., 1.,
                       0., 0., 0.], rtol=0, atol=1e-15)
    xo.assert_allclose(line['v1'].ksl, [4., 5., 6.,
                       0., 0., 0.], rtol=0, atol=1e-15)
    xo.assert_allclose(line['m1'].knl, [3., 2., 1.], rtol=0, atol=1e-15)
    xo.assert_allclose(line['m1'].ksl, [4., 5., 6.], rtol=0, atol=1e-15)

    line.extend_knl_ksl(order=11)

    assert line['b1'].order == 11
    assert line['q1'].order == 11
    assert line['s1'].order == 11
    assert line['o1'].order == 11
    assert line['u1'].order == 11
    assert line['v1'].order == 11
    assert line['m1'].order == 11
    assert line['b1'].inv_factorial_order == 1/math.factorial(11)
    assert line['q1'].inv_factorial_order == 1/math.factorial(11)
    assert line['s1'].inv_factorial_order == 1/math.factorial(11)
    assert line['o1'].inv_factorial_order == 1/math.factorial(11)
    assert line['u1'].inv_factorial_order == 1/math.factorial(11)
    assert line['v1'].inv_factorial_order == 1/math.factorial(11)
    assert line['m1'].inv_factorial_order == 1/math.factorial(11)
    xo.assert_allclose(line['b1'].knl, [3., 2., 1., 0.,
                       0., 0., 0., 0., 0., 0., 0., 0.], rtol=0, atol=1e-15)
    xo.assert_allclose(line['b1'].ksl, [4., 5., 6., 0.,
                       0., 0., 0., 0., 0., 0., 0., 0.], rtol=0, atol=1e-15)
    xo.assert_allclose(line['q1'].knl, [3., 2., 1., 0.,
                       0., 0., 0., 0., 0., 0., 0., 0.], rtol=0, atol=1e-15)
    xo.assert_allclose(line['q1'].ksl, [4., 5., 6., 0.,
                       0., 0., 0., 0., 0., 0., 0., 0.], rtol=0, atol=1e-15)
    xo.assert_allclose(line['s1'].knl, [3., 2., 1., 0.,
                       0., 0., 0., 0., 0., 0., 0., 0.], rtol=0, atol=1e-15)
    xo.assert_allclose(line['s1'].ksl, [4., 5., 6., 0.,
                       0., 0., 0., 0., 0., 0., 0., 0.], rtol=0, atol=1e-15)
    xo.assert_allclose(line['o1'].knl, [3., 2., 1., 0.,
                       0., 0., 0., 0., 0., 0., 0., 0.], rtol=0, atol=1e-15)
    xo.assert_allclose(line['o1'].ksl, [4., 5., 6., 0.,
                       0., 0., 0., 0., 0., 0., 0., 0.], rtol=0, atol=1e-15)
    xo.assert_allclose(line['u1'].knl, [3., 2., 1., 0.,
                       0., 0., 0., 0., 0., 0., 0., 0.], rtol=0, atol=1e-15)
    xo.assert_allclose(line['v1'].ksl, [4., 5., 6., 0.,
                       0., 0., 0., 0., 0., 0., 0., 0.], rtol=0, atol=1e-15)
    xo.assert_allclose(line['v1'].knl, [3., 2., 1., 0.,
                       0., 0., 0., 0., 0., 0., 0., 0.], rtol=0, atol=1e-15)
    xo.assert_allclose(line['u1'].ksl, [4., 5., 6., 0.,
                       0., 0., 0., 0., 0., 0., 0., 0.], rtol=0, atol=1e-15)
    xo.assert_allclose(line['m1'].knl, [3., 2., 1., 0.,
                       0., 0., 0., 0., 0., 0., 0., 0.], rtol=0, atol=1e-15)
    xo.assert_allclose(line['m1'].ksl, [4., 5., 6., 0.,
                       0., 0., 0., 0., 0., 0., 0., 0.], rtol=0, atol=1e-15)

    # test an expression
    line['b'] = 100
    line['f'] = 200

    xo.assert_allclose(line['o1'].knl, [3., 100., 1., 0.,
                       0., 0., 0., 0., 0., 0., 0., 0.], rtol=0, atol=1e-15)
    xo.assert_allclose(line['o1'].ksl, [4., 5., 200., 0.,
                       0., 0., 0., 0., 0., 0., 0., 0.], rtol=0, atol=1e-15)


