# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import xobjects as xo
import xtrack as xt


def test_tracker_data_serialization():
    line = xt.Line(
        elements={
            'mn': xt.Multipole(knl=[1, 2]),
            'ms': xt.Multipole(ksl=[3]),
            'd': xt.Drift(length=4),
        },
        element_names=['mn', 'd', 'ms', 'd', 'mn'],
    )

    tracker_data = xt.TrackerData(line=line)

    buffer, header_offset = tracker_data.to_binary()
    new_tracker_data = tracker_data.from_binary(buffer, header_offset)

    assert tracker_data.element_names == new_tracker_data.element_names

    assert [elem.__class__.__name__ for elem in tracker_data.elements] == \
           ['Multipole', 'Drift', 'Multipole', 'Drift', 'Multipole']
    assert new_tracker_data.elements[0]._xobject._offset == \
           new_tracker_data.elements[4]._xobject._offset
    assert new_tracker_data.elements[1]._xobject._offset == \
           new_tracker_data.elements[3]._xobject._offset

    assert len(set(elem._xobject._buffer for elem in new_tracker_data.elements)) == 1

    assert (new_tracker_data.elements[0].knl == [1, 2]).all()
    assert new_tracker_data.elements[1].length == 4
    assert (new_tracker_data.elements[2].ksl == [3]).all()


def test_tracker_data_serialization_same_buffer():
    buffer = xo.context_default.new_buffer(0)
    line = xt.Line(
        elements={
            'mn': xt.Multipole(knl=[1, 2], _buffer=buffer),
            'ms': xt.Multipole(ksl=[3], _buffer=buffer),
            'd': xt.Drift(length=4, _buffer=buffer),
        },
        element_names=['mn', 'd', 'ms', 'd', 'mn'],
    )

    tracker_data = xt.TrackerData(line=line)

    assert tracker_data._buffer is buffer

    out_buffer, header_offset = tracker_data.to_binary()
    new_tracker_data = tracker_data.from_binary(out_buffer, header_offset)

    assert tracker_data.element_names == new_tracker_data.element_names

    assert [elem.__class__.__name__ for elem in tracker_data.elements] == \
           ['Multipole', 'Drift', 'Multipole', 'Drift', 'Multipole']
    assert new_tracker_data.elements[0]._xobject._offset == \
           new_tracker_data.elements[4]._xobject._offset
    assert new_tracker_data.elements[1]._xobject._offset == \
           new_tracker_data.elements[3]._xobject._offset

    assert len(set(elem._xobject._buffer for elem in new_tracker_data.elements)) == 1

    assert (new_tracker_data.elements[0].knl == [1, 2]).all()
    assert new_tracker_data.elements[1].length == 4
    assert (new_tracker_data.elements[2].ksl == [3]).all()


def test_tracker_data_serialization_across_contexts():
    buffer = xo.ContextCpu().new_buffer(0)
    line = xt.Line(
        elements={
            'mn': xt.Multipole(knl=[1, 2], _buffer=buffer),
            'ms': xt.Multipole(ksl=[3], _buffer=buffer),
            'd': xt.Drift(length=4, _buffer=buffer),
        },
        element_names=['mn', 'd', 'ms', 'd', 'mn'],
    )

    fresh_buffer = xo.ContextCpu().new_buffer(0)

    tracker_data = xt.TrackerData(line=line, _buffer=fresh_buffer)

    assert tracker_data._buffer is not buffer
    assert tracker_data._buffer is fresh_buffer

    out_buffer, header_offset = tracker_data.to_binary()
    new_tracker_data = tracker_data.from_binary(out_buffer, header_offset)

    assert out_buffer is fresh_buffer
    assert tracker_data.element_names == new_tracker_data.element_names

    assert [elem.__class__.__name__ for elem in tracker_data.elements] == \
           ['Multipole', 'Drift', 'Multipole', 'Drift', 'Multipole']
    assert new_tracker_data.elements[0]._xobject._offset == \
           new_tracker_data.elements[4]._xobject._offset
    assert new_tracker_data.elements[1]._xobject._offset == \
           new_tracker_data.elements[3]._xobject._offset

    assert len(set(el._xobject._buffer for el in new_tracker_data.elements)) == 1

    assert (new_tracker_data.elements[0].knl == [1, 2]).all()
    assert new_tracker_data.elements[1].length == 4
    assert (new_tracker_data.elements[2].ksl == [3]).all()


def test_tracker_data_serialization_into_new_buffer():
    line = xt.Line(
        elements={
            'mn': xt.Multipole(knl=[1, 2]),
            'ms': xt.Multipole(ksl=[3]),
            'd': xt.Drift(length=4),
        },
        element_names=['mn', 'd', 'ms', 'd', 'mn'],
    )

    tracker_data = xt.TrackerData(line=line)
    fresh_buffer = xo.ContextCpu().new_buffer(0)

    out_buffer, header_offset = tracker_data.to_binary(buffer=fresh_buffer)
    new_tracker_data = tracker_data.from_binary(out_buffer, header_offset)

    assert out_buffer is fresh_buffer
    assert tracker_data.element_names == new_tracker_data.element_names

    assert [elem.__class__.__name__ for elem in tracker_data.elements] == \
           ['Multipole', 'Drift', 'Multipole', 'Drift', 'Multipole']
    assert new_tracker_data.elements[0]._xobject._offset == \
           new_tracker_data.elements[4]._xobject._offset
    assert new_tracker_data.elements[1]._xobject._offset == \
           new_tracker_data.elements[3]._xobject._offset

    assert len(set(el._xobject._buffer for el in new_tracker_data.elements)) == 1

    assert (new_tracker_data.elements[0].knl == [1, 2]).all()
    assert new_tracker_data.elements[1].length == 4
    assert (new_tracker_data.elements[2].ksl == [3]).all()
