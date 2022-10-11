# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np

import xobjects
import xtrack as xt

def test_simplification_methods():

    line = xt.Line(
        elements=([xt.Drift(length=0)] # Start line marker
                    + [xt.Drift(length=1) for _ in range(5)]
                    + [xt.Drift(length=0)] # End line marker
            )
        )

    line.insert_element(element=xt.Cavity(), name="cav", at_s=3.3)
    line.merge_consecutive_drifts(inplace=True)
    assert len(line.element_names) == 3
    assert line.get_length() == line.get_s_elements(mode='downstream')[-1] == 5
    assert np.isclose(line[0].length, 3.3, rtol=0, atol=1e-12)
    assert isinstance(line[1], xt.Cavity)
    assert np.isclose(line[2].length, 1.7, rtol=0, atol=1e-12)

    line.insert_element(element=xt.Drift(length=0), name="marker", at_s=3.3)
    assert len(line.element_names) == 4
    line.remove_zero_length_drifts(inplace=True)
    assert len(line.element_names) == 3

    line.insert_element(element=xt.Multipole(knl=[1, 0, 3], ksl=[0, 20, 0]), name="m1", at_s=3.3)
    line.insert_element(element=xt.Multipole(knl=[4, 2], ksl=[10, 40]), name="m2", at_s=3.3)
    assert len(line.element_names) == 5
    line.merge_consecutive_multipoles(inplace=True)
    assert len(line.element_names) == 4
    assert np.allclose(line[1].knl, [5,2,3], rtol=0, atol=1e-15)
    assert np.allclose(line[1].ksl, [10,60,0], rtol=0, atol=1e-15)

    line.remove_inactive_multipoles(inplace=True)
    assert len(line.element_names) == 4
    line[1].knl[:] = 0
    line[1].ksl[:] = 0
    line.remove_inactive_multipoles(inplace=True)
    assert len(line.element_names) == 3

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


def test_line_frozen_serialization():
    line = xt.Line(
        elements={
            'mn': xt.Multipole(knl=[1, 2]),
            'ms': xt.Multipole(ksl=[3]),
            'd': xt.Drift(length=4),
        },
        element_names=['mn', 'd', 'ms', 'd', 'mn'],
    )

    frozen = xt.LineFrozen(line=line)

    buffer, header_offset = frozen.serialize()
    new_frozen = frozen.deserialize(buffer, header_offset)

    assert frozen.element_names == new_frozen.element_names

    assert [elem.__class__.__name__ for elem in frozen.elements] == \
           ['Multipole', 'Drift', 'Multipole', 'Drift', 'Multipole']
    assert new_frozen.elements[0]._xobject._offset == \
           new_frozen.elements[4]._xobject._offset
    assert new_frozen.elements[1]._xobject._offset == \
           new_frozen.elements[3]._xobject._offset

    assert len(set(elem._xobject._buffer for elem in new_frozen.elements)) == 1

    assert (new_frozen.elements[0].knl == [1, 2]).all()
    assert new_frozen.elements[1].length == 4
    assert (new_frozen.elements[2].ksl == [3]).all()


def test_line_frozen_serialization_same_buffer():
    buffer = xobjects.context_default.new_buffer(0)
    line = xt.Line(
        elements={
            'mn': xt.Multipole(knl=[1, 2], _buffer=buffer),
            'ms': xt.Multipole(ksl=[3], _buffer=buffer),
            'd': xt.Drift(length=4, _buffer=buffer),
        },
        element_names=['mn', 'd', 'ms', 'd', 'mn'],
    )

    frozen = xt.LineFrozen(line=line)

    assert frozen._buffer is buffer

    out_buffer, header_offset = frozen.serialize()
    new_frozen = frozen.deserialize(out_buffer, header_offset)

    assert frozen.element_names == new_frozen.element_names

    assert [elem.__class__.__name__ for elem in frozen.elements] == \
           ['Multipole', 'Drift', 'Multipole', 'Drift', 'Multipole']
    assert new_frozen.elements[0]._xobject._offset == \
           new_frozen.elements[4]._xobject._offset
    assert new_frozen.elements[1]._xobject._offset == \
           new_frozen.elements[3]._xobject._offset

    assert len(set(elem._xobject._buffer for elem in new_frozen.elements)) == 1

    assert (new_frozen.elements[0].knl == [1, 2]).all()
    assert new_frozen.elements[1].length == 4
    assert (new_frozen.elements[2].ksl == [3]).all()


def test_line_frozen_serialization_across_contexts():
    buffer = xobjects.ContextCpu().new_buffer(0)
    line = xt.Line(
        elements={
            'mn': xt.Multipole(knl=[1, 2], _buffer=buffer),
            'ms': xt.Multipole(ksl=[3], _buffer=buffer),
            'd': xt.Drift(length=4, _buffer=buffer),
        },
        element_names=['mn', 'd', 'ms', 'd', 'mn'],
    )

    fresh_buffer = xobjects.ContextCpu().new_buffer(0)

    frozen = xt.LineFrozen(line=line, _buffer=fresh_buffer)

    assert frozen._buffer is not buffer
    assert frozen._buffer is fresh_buffer

    out_buffer, header_offset = frozen.serialize()
    new_frozen = frozen.deserialize(out_buffer, header_offset)

    assert out_buffer is fresh_buffer
    assert frozen.element_names == new_frozen.element_names

    assert [elem.__class__.__name__ for elem in frozen.elements] == \
           ['Multipole', 'Drift', 'Multipole', 'Drift', 'Multipole']
    assert new_frozen.elements[0]._xobject._offset == \
           new_frozen.elements[4]._xobject._offset
    assert new_frozen.elements[1]._xobject._offset == \
           new_frozen.elements[3]._xobject._offset

    assert len(set(elem._xobject._buffer for elem in new_frozen.elements)) == 1

    assert (new_frozen.elements[0].knl == [1, 2]).all()
    assert new_frozen.elements[1].length == 4
    assert (new_frozen.elements[2].ksl == [3]).all()



def test_line_frozen_serialization_into_new_buffer():
    line = xt.Line(
        elements={
            'mn': xt.Multipole(knl=[1, 2]),
            'ms': xt.Multipole(ksl=[3]),
            'd': xt.Drift(length=4),
        },
        element_names=['mn', 'd', 'ms', 'd', 'mn'],
    )

    frozen = xt.LineFrozen(line=line)
    fresh_buffer = xobjects.ContextCpu().new_buffer(0)

    out_buffer, header_offset = frozen.serialize(buffer=fresh_buffer)
    new_frozen = frozen.deserialize(out_buffer, header_offset)

    assert out_buffer is fresh_buffer
    assert frozen.element_names == new_frozen.element_names

    assert [elem.__class__.__name__ for elem in frozen.elements] == \
           ['Multipole', 'Drift', 'Multipole', 'Drift', 'Multipole']
    assert new_frozen.elements[0]._xobject._offset == \
           new_frozen.elements[4]._xobject._offset
    assert new_frozen.elements[1]._xobject._offset == \
           new_frozen.elements[3]._xobject._offset

    assert len(set(elem._xobject._buffer for elem in new_frozen.elements)) == 1

    assert (new_frozen.elements[0].knl == [1, 2]).all()
    assert new_frozen.elements[1].length == 4
    assert (new_frozen.elements[2].ksl == [3]).all()

