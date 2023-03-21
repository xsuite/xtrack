# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np

import xtrack as xt
from xtrack import Line, Node, Multipole
from xobjects.test_helpers import for_all_test_contexts


def test_simplification_methods():

    line = xt.Line(
        elements=([xt.Drift(length=0)] # Start line marker
                    + [xt.Drift(length=1) for _ in range(5)]
                    + [xt.Drift(length=0)] # End line marker
            )
        )

    # Test merging of drifts
    line.insert_element(element=xt.Cavity(), name='cav', at_s=3.3)
    line.merge_consecutive_drifts(inplace=True)
    assert len(line.element_names) == 3
    assert line.get_length() == line.get_s_elements(mode='downstream')[-1] == 5
    assert np.isclose(line[0].length, 3.3, rtol=0, atol=1e-12)
    assert isinstance(line[1], xt.Cavity)
    assert np.isclose(line[2].length, 1.7, rtol=0, atol=1e-12)

    # Test merging of drifts, while keeping one
    line.insert_element(element=xt.Drift(length=1), name='drift1', at_s=1.2)
    line.insert_element(element=xt.Drift(length=1), name='drift2', at_s=2.2)
    line.merge_consecutive_drifts(inplace=True, keep=['drift2'])
    assert len(line.element_names) == 5
    assert 'drift2' in line.element_names
    assert 'drift1' not in line.element_names
    line.merge_consecutive_drifts(inplace=True)

    # Test removing of zero-length drifts
    line.insert_element(element=xt.Drift(length=0), name='marker1', at_s=3.3)
    line.insert_element(element=xt.Drift(length=0), name='marker2', at_s=3.3)
    assert len(line.element_names) == 5
    line.remove_zero_length_drifts(inplace=True, keep='marker2')
    assert len(line.element_names) == 4
    assert 'marker2' in line.element_names
    assert 'marker1' not in line.element_names
    line.remove_zero_length_drifts(inplace=True)

    # Test merging of multipoles
    line._var_management = None
    line.insert_element(element=xt.Multipole(knl=[1, 0, 3], ksl=[0, 20, 0]), name='m1', at_s=3.3)
    line.insert_element(element=xt.Multipole(knl=[4, 2], ksl=[10, 40]), name='m2', at_s=3.3)
    line.insert_element(element=xt.Multipole(knl=[0, 3, 8], ksl=[2, 0, 17]), name='m3', at_s=3.3)
    line.insert_element(element=xt.Multipole(knl=[2, 0, 0], ksl=[40]), name='m4', at_s=3.3)
    assert len(line.element_names) == 7
    line.merge_consecutive_multipoles(inplace=True, keep='m3')
    assert len(line.element_names) == 6
    assert 'm3' in line.element_names
    # We merged the first two multipoles
    joined_mult = [ name for name in line.element_names if 'm1' in name]
    assert len(joined_mult) == 1
    joined_mult = joined_mult[0]
    assert 'm2' in joined_mult
    assert np.allclose(line[joined_mult].knl, [5,2,3], rtol=0, atol=1e-15)
    assert np.allclose(line[joined_mult].ksl, [10,60,0], rtol=0, atol=1e-15)
    # Merging all
    line.merge_consecutive_multipoles(inplace=True)
    assert len(line.element_names) == 4
    assert np.allclose(line[1].knl, [7,5,11], rtol=0, atol=1e-15)
    assert np.allclose(line[1].ksl, [52,60,17], rtol=0, atol=1e-15)

    # Test removing inactive multipoles
    line.insert_element(element=xt.Multipole(knl=[0, 8, 1], ksl=[0, 20, 30]), name='m5', at_s=3.3)
    line.insert_element(element=xt.Multipole(knl=[2, 0, 3], ksl=[10, 34, 15]), name='m6', at_s=3.3)
    line.remove_inactive_multipoles(inplace=True)
    assert len(line.element_names) == 6
    line['m5'].knl[:] = 0
    line['m5'].ksl[:] = 0
    line['m6'].knl[:] = 0
    line['m6'].ksl[:] = 0
    line.remove_inactive_multipoles(inplace=True, keep='m5')
    assert len(line.element_names) == 5
    assert 'm5' in line.element_names
    assert 'm6' not in line.element_names

    # Test removing markers
    line.insert_element(element=xt.Marker(), name='marker1', at_s=3.3)
    line.insert_element(element=xt.Marker(), name='marker2', at_s=3.3)
    assert 'marker1' in line.element_names
    assert 'marker2' in line.element_names
    line.remove_markers(keep='marker2')
    assert 'marker1' not in line.element_names
    assert 'marker2' in line.element_names
    line.insert_element(element=xt.Marker(), name='marker4', at_s=3.3)
    line.insert_element(element=xt.Marker(), name='marker3', at_s=3.3)
    assert 'marker2' in line.element_names
    assert 'marker3' in line.element_names
    assert 'marker4' in line.element_names
    line.remove_markers()
    assert 'marker2' not in line.element_names
    assert 'marker3' not in line.element_names
    assert 'marker4' not in line.element_names


def test_simplification_methods_not_inplace():

    line = xt.Line(
        elements=([xt.Drift(length=0)] # Start line marker
                    + [xt.Drift(length=1) for _ in range(5)]
                    + [xt.Drift(length=0)] # End line marker
            )
        )

    # Test merging of drifts
    line.insert_element(element=xt.Cavity(), name="cav", at_s=3.3)
    original_line = line.copy()
    newline = line.merge_consecutive_drifts(inplace=False)
    assert xt._lines_equal(line, original_line)
    assert len(newline.element_names) == 3
    assert newline.get_length() == newline.get_s_elements(mode='downstream')[-1] == 5
    assert np.isclose(newline[0].length, 3.3, rtol=0, atol=1e-12)
    assert isinstance(newline[1], xt.Cavity)
    assert np.isclose(newline[2].length, 1.7, rtol=0, atol=1e-12)
    line.merge_consecutive_drifts(inplace=True)

    # Test removing of zero-length drifts
    line.insert_element(element=xt.Drift(length=0), name="marker", at_s=3.3)
    assert len(line.element_names) == 4
    original_line = line.copy()
    newline = line.remove_zero_length_drifts(inplace=False)
    assert xt._lines_equal(line, original_line)
    assert len(newline.element_names) == 3
    line.remove_zero_length_drifts(inplace=True)

    # Test merging of multipoles
    line._var_management = None
    line.insert_element(element=xt.Multipole(knl=[1, 0, 3], ksl=[0, 20, 0]), name="m1", at_s=3.3)
    line.insert_element(element=xt.Multipole(knl=[4, 2], ksl=[10, 40]), name="m2", at_s=3.3)
    assert len(line.element_names) == 5
    original_line = line.copy()
    newline = line.merge_consecutive_multipoles(inplace=False)
    assert xt._lines_equal(line, original_line)
    assert len(newline.element_names) == 4
    assert np.allclose(newline[1].knl, [5,2,3], rtol=0, atol=1e-15)
    assert np.allclose(newline[1].ksl, [10,60,0], rtol=0, atol=1e-15)
    line.merge_consecutive_multipoles(inplace=True)

    # Test removing inactive multipoles
    original_line = line.copy()
    newline = line.remove_inactive_multipoles(inplace=False)
    assert xt._lines_equal(line, original_line)
    assert len(newline.element_names) == 4
    line.remove_inactive_multipoles(inplace=True)

    line[1].knl[:] = 0
    line[1].ksl[:] = 0
    original_line = line.copy()
    newline = line.remove_inactive_multipoles(inplace=False)
    assert xt._lines_equal(line, original_line)
    assert len(newline.element_names) == 3
    line.remove_inactive_multipoles(inplace=True)

    # Test removing markers
    line.insert_element(element=xt.Marker(), name='marker1', at_s=3.3)
    line.insert_element(element=xt.Marker(), name='marker2', at_s=3.3)
    assert 'marker1' in line.element_names
    assert 'marker2' in line.element_names
    original_line = line.copy()
    newline = line.remove_markers(inplace=False, keep='marker2')
    assert xt._lines_equal(line, original_line)
    assert 'marker1' not in newline.element_names
    assert 'marker2' in newline.element_names
    line.remove_markers(inplace=True, keep='marker2')

    line.insert_element(element=xt.Marker(), name='marker4', at_s=3.3)
    line.insert_element(element=xt.Marker(), name='marker3', at_s=3.3)
    assert 'marker2' in line.element_names
    assert 'marker3' in line.element_names
    assert 'marker4' in line.element_names
    original_line = line.copy()
    newline = line.remove_markers(inplace=False)
    assert xt._lines_equal(line, original_line)
    assert 'marker2' not in newline.element_names
    assert 'marker3' not in newline.element_names
    assert 'marker4' not in newline.element_names


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
    all_aper = [nn for nn in line.element_names if xt._is_aperture(line[nn])]
    all_aper_pos = [line.get_s_position(ap) for ap in all_aper]
    line.remove_redundant_apertures()
    line.remove_markers()
    line.merge_consecutive_drifts()
    # The lattice is now D1-A1-DD-A5-D10
    assert len(line.element_names) == 5
    # Verify that only the first and last aperture are kept
    new_aper = [nn for nn in line.element_names if xt._is_aperture(line[nn])]
    assert new_aper == [all_aper[0], all_aper[-1]]
    new_aper_pos = [line.get_s_position(ap) for ap in new_aper]
    assert new_aper_pos == [all_aper_pos[0], all_aper_pos[-1]]

    # Test removing all consecutive middle apertures, but
    # keep the 4th one (and hence also the 5th)
    line = original_line.copy()
    assert len(line.element_names) == 20
    all_aper = [nn for nn in line.element_names if xt._is_aperture(line[nn])]
    all_aper_pos = [line.get_s_position(ap) for ap in all_aper]
    line.remove_redundant_apertures(keep=all_aper[3])
    line.remove_markers()
    line.merge_consecutive_drifts()
    # The lattice is now D1-A1-DD-A4-DD-A5-D10
    assert len(line.element_names) == 7
    # Verify that only the first, fourth, and last aperture are kept
    new_aper = [nn for nn in line.element_names if xt._is_aperture(line[nn])]
    assert new_aper == [all_aper[0], all_aper[3], all_aper[-1]]
    new_aper_pos = [line.get_s_position(ap) for ap in new_aper]
    assert new_aper_pos == [all_aper_pos[0], all_aper_pos[3], all_aper_pos[-1]]

    # Test removing all consecutive middle apertures, but
    # the 9th Drift needs to have an aperture. This should
    # give the same result as above
    line = original_line.copy()
    assert len(line.element_names) == 20
    all_aper = [nn for nn in line.element_names if xt._is_aperture(line[nn])]
    all_aper_pos = [line.get_s_position(ap) for ap in all_aper]
    all_drifts = [nn for nn in line.element_names if xt._is_drift(line[nn])]
    line.remove_redundant_apertures(drifts_that_need_aperture=all_drifts[8])
    line.remove_markers()
    line.merge_consecutive_drifts()
    # The lattice is now D1-A1-DD-A4-DD-A5-D10
    assert len(line.element_names) == 7
    # Verify that only the first, fourth, and last aperture are kept
    new_aper = [nn for nn in line.element_names if xt._is_aperture(line[nn])]
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

    
def test_remove_redundant_apertures_not_inplace():

    # Test removing all consecutive middle apertures
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

    assert len(line.element_names) == 20
    all_aper = [nn for nn in line.element_names if xt._is_aperture(line[nn])]
    all_aper_pos = [line.get_s_position(ap) for ap in all_aper]
    newline = line.remove_redundant_apertures(inplace=False)
    newline = newline.remove_markers(inplace=False)
    newline = newline.merge_consecutive_drifts(inplace=False)
    assert xt._lines_equal(line, original_line)

    assert len(newline.element_names) == 5
    # Verify that only the first and last aperture are kept
    new_aper = [nn for nn in newline.element_names if xt._is_aperture(newline[nn])]
    assert new_aper == [all_aper[0], all_aper[-1]]
    new_aper_pos = [newline.get_s_position(ap) for ap in new_aper]
    assert new_aper_pos == [all_aper_pos[0], all_aper_pos[-1]]


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
                ['e0_part0', 'inserted_drift', 'e0_part1', 'e1', 'e2', 'e3', 'e4']))])
    assert line.get_length() == line.get_s_elements(mode='downstream')[-1] == 5

    line = line0.copy()
    line.insert_element(element=xt.Drift(length=0.2), at_s=0.95, name='inserted_drift')
    assert line.get_s_position('inserted_drift') == 0.95
    assert len(line.elements) == 6
    assert np.all([nn==nnref for nn, nnref in list(zip(line.element_names,
                ['e0_part0', 'inserted_drift', 'e1_part1', 'e2', 'e3', 'e4']))])
    assert line.get_length() == line.get_s_elements(mode='downstream')[-1] == 5

    line = line0.copy()
    line.insert_element(element=xt.Drift(length=0.2), at_s=1.0, name='inserted_drift')
    assert line.get_s_position('inserted_drift') == 1.
    assert len(line.elements) == 6
    assert np.all([nn==nnref for nn, nnref in list(zip(line.element_names,
                ['e0', 'inserted_drift', 'e1_part1', 'e2', 'e3', 'e4']))])
    assert line.get_length() == line.get_s_elements(mode='downstream')[-1] == 5

    line = line0.copy()
    line.insert_element(element=xt.Drift(length=0.2), at_s=0.8, name='inserted_drift')
    assert line.get_s_position('inserted_drift') == 0.8
    assert len(line.elements) == 6
    assert np.all([nn==nnref for nn, nnref in list(zip(line.element_names,
                ['e0_part0', 'inserted_drift', 'e1', 'e2', 'e3', 'e4']))])
    assert line.get_length() == line.get_s_elements(mode='downstream')[-1] == 5

    line = line0.copy()
    line.insert_element(element=xt.LimitEllipse(a=1, b=1), at_s=2.1, name='aper')
    assert line.get_s_position('aper') == 2.1
    assert line.get_length() == line.get_s_elements(mode='downstream')[-1] == 5
    assert np.all([nn==nnref for nn, nnref in list(zip(line.element_names,
                ['e0', 'e1', 'e2_part0', 'aper', 'e2_part1', 'e3', 'e4']))])
    line.insert_element(element=xt.Drift(length=0.8), at_s=1.9, name="newdrift")
    assert line.get_s_position('newdrift') == 1.9
    assert np.all([nn==nnref for nn, nnref in list(zip(line.element_names,
                ['e0', 'e1_part0', 'newdrift', 'e2_part1_part1', 'e3', 'e4']))])

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
                ['d0_part0', 'inserted_drift', 'd1_part1', 'm1', 'd2', 'm2', 'd3',
                'm3', 'd4', 'm4']))])
    assert line.get_length() == line.get_s_elements(mode='downstream')[-1] == 5

def test_to_pandas():

    line = xt.Line(elements=[
        xt.Drift(length=1), xt.Cavity(), xt.Drift(length=1)])

    df = line.to_pandas()

    assert tuple(df.columns) == (
                            'element_type', 's', 'name', 'isthick', 'element')
    assert len(df) == 3

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
        },
        element_names=['dr1', 'm1_ap', 'dum', 'm1', 'dr2', 'm2', 'dr3',
                       'th1_ap_front', 'dum', 'th1', 'dum', 'th1_ap_back',
                       'dr4', 'th2', 'th2_ap_back',
                       'dr5', 'th3_ap_front', 'th3'])
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
        'element_names': ['mn', 'd', 'ms', 'd']
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
        assert np.isclose(line.elements[i].length, l)

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
        assert np.isclose(line.elements[i].length, l)
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
    try:
        Line.from_sequence([Node(3, Multipole()), Node(2, Multipole())], 10)
    except ValueError:
        pass  # expected due to negative drift
    else:
        raise AssertionError('Expected exception not raised')
    try:
        Line.from_sequence([Node(1, Multipole()), Node(4, Multipole())], 2)
    except ValueError:
        pass  # expected due to insufficient length
    else:
        raise AssertionError('Expected exception not raised')

@for_all_test_contexts
def test_optimize_multipoles(test_context):
    elements = {
        'q1': xt.Multipole(knl=[0, 1], length=0, _context=test_context),
        'q2': xt.Multipole(knl=[1, 2], length=0, _context=test_context),
        'q3': xt.Multipole(knl=[0, 1], length=1, _context=test_context),
        'q4': xt.Multipole(knl=[0, 1], ksl=[1, 2], length=0, _context=test_context),
        'd1': xt.Multipole(knl=[1], hxl=0.0, length=2, _context=test_context),
        'd2': xt.Multipole(knl=[1], hxl=0.1, length=0, _context=test_context),
        'd3': xt.Multipole(knl=[1], hyl=1, length=2, _context=test_context),
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
        elif nn == 'q1':
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

    line.to_json(tmp_path / 'test.json')
    result = xt.Line.from_json(tmp_path / 'test.json')

    assert len(result.element_dict.keys()) == 2
    assert result.element_names == ['m', 'd', 'm', 'd']

    assert isinstance(result['m'], xt.Multipole)
    assert (result['m'].knl == [1, 2]).all()

    assert isinstance(result['d'], xt.Drift)
    assert result['d'].length == 1

    with open(tmp_path / 'test2.json', 'w') as f:
        line.to_json(f)

    with open(tmp_path / 'test2.json', 'r') as f:
        result = xt.Line.from_json(f)

    assert len(result.element_dict.keys()) == 2
    assert result.element_names == ['m', 'd', 'm', 'd']

    assert isinstance(result['m'], xt.Multipole)
    assert (result['m'].knl == [1, 2]).all()

    assert isinstance(result['d'], xt.Drift)
    assert result['d'].length == 1
