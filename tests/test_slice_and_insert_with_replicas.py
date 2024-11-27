import xtrack as xt
import xobjects as xo
import numpy as np
from xobjects.test_helpers import for_all_test_contexts

assert_allclose = xo.assert_allclose

@for_all_test_contexts
def test_slice_thin_and_insert_with_replicas(test_context):

    elements = {
        'e0': xt.Bend(k0=0.3, h=0.31, length=1),
        'e1': xt.Replica(parent_name='e0'),
        'e2': xt.Bend(k0=-0.4, h=-0.41, length=1),
        'e3': xt.Replica(parent_name='e2'),
        'e4': xt.Replica(parent_name='e3'), # Replica of replica
    }
    line = xt.Line(elements=elements,
                element_names=list(elements.keys()))
    line.build_tracker(_context=test_context)

    element_no_repl={
        'e0': xt.Bend(k0=0.3, h=0.31, length=1),
        'e1': xt.Bend(k0=0.3, h=0.31, length=1),
        'e2': xt.Bend(k0=-0.4, h=-0.41, length=1),
        'e3': xt.Bend(k0=-0.4, h=-0.41, length=1),
        'e4': xt.Bend(k0=-0.4, h=-0.41, length=1),
    }

    line_no_repl = xt.Line(elements=element_no_repl,
                        element_names=list(element_no_repl.keys()))
    line_no_repl.build_tracker(_context=test_context)

    tt = line.get_table()
    tt_no_repl = line_no_repl.get_table()

    assert np.all(tt.name == ['e0', 'e1', 'e2', 'e3', 'e4', '_end_point'])
    assert np.all(tt_no_repl.name == ['e0', 'e1', 'e2', 'e3', 'e4', '_end_point'])

    assert np.all(tt.parent_name == [None, 'e0', None, 'e2', 'e3', None])
    assert np.all(tt_no_repl.parent_name == [None, None, None, None, None, None])

    assert_allclose(tt.s, [0., 1., 2., 3., 4., 5.], rtol=0, atol=1e-14)
    assert_allclose(tt_no_repl.s, [0., 1., 2., 3., 4., 5.], rtol=0, atol=1e-14)

    p0 = xt.Particles(p0c=1e9, x=0.1, px=0.2, y=0.3, py=0.4, delta=0.03,
                      _context=test_context)
    p1 = p0.copy()
    p2 = p0.copy()

    line.track(p1)
    line_no_repl.track(p2)

    assert_allclose(p2.x, p1.x, rtol=0, atol=1e-14)
    assert_allclose(p2.px, p1.px, rtol=0, atol=1e-14)
    assert_allclose(p2.y, p1.y, rtol=0, atol=1e-14)
    assert_allclose(p2.py, p1.py, rtol=0, atol=1e-14)
    assert_allclose(p2.zeta, p1.zeta, rtol=0, atol=1e-14)
    assert_allclose(p2.delta, p1.delta, rtol=0, atol=1e-14)

    line.slice_thick_elements(
        slicing_strategies=[
            xt.Strategy(None),
            xt.Strategy(xt.Teapot(2, mode='thin'), name='e2|e3|e4')])
    line.build_tracker(_context=test_context)

    line_no_repl.slice_thick_elements(
        slicing_strategies=[
            xt.Strategy(None),
            xt.Strategy(xt.Teapot(2, mode='thin'), name='e2|e3|e4')])
    line_no_repl.build_tracker(_context=test_context)

    tt = line.get_table()
    tt_no_repl = line_no_repl.get_table()

    assert np.all(tt.name == np.array(['e0', 'e1', 'e2_entry', 'e2..entry_map', 'drift_e2..0', 'e2..0',
        'drift_e2..1', 'e2..1', 'drift_e2..2', 'e2..exit_map', 'e2_exit',
        'e3_entry', 'e3..entry_map', 'drift_e3..0', 'e3..0', 'drift_e3..1',
        'e3..1', 'drift_e3..2', 'e3..exit_map', 'e3_exit', 'e4_entry',
        'e4..entry_map', 'drift_e4..0', 'e4..0', 'drift_e4..1', 'e4..1',
        'drift_e4..2', 'e4..exit_map', 'e4_exit', '_end_point']))

    assert np.all(tt_no_repl.name == np.array([
        'e0', 'e1', 'e2_entry', 'e2..entry_map', 'drift_e2..0', 'e2..0',
        'drift_e2..1', 'e2..1', 'drift_e2..2', 'e2..exit_map', 'e2_exit',
        'e3_entry', 'e3..entry_map', 'drift_e3..0', 'e3..0', 'drift_e3..1',
        'e3..1', 'drift_e3..2', 'e3..exit_map', 'e3_exit', 'e4_entry',
        'e4..entry_map', 'drift_e4..0', 'e4..0', 'drift_e4..1', 'e4..1',
        'drift_e4..2', 'e4..exit_map', 'e4_exit', '_end_point']))

    assert np.all(tt.parent_name == np.array([
        None, 'e0', None, 'e2', 'e2', 'e2', 'e2', 'e2', 'e2', 'e2', None,
        None, 'e2', 'e2', 'e2', 'e2', 'e2', 'e2', 'e2', None, None, 'e2',
        'e2', 'e2', 'e2', 'e2', 'e2', 'e2', None, None]))
    assert np.all(tt_no_repl.parent_name == np.array([None, None, None,
        'e2', 'e2', 'e2', 'e2', 'e2', 'e2', 'e2', None,
        None, 'e3', 'e3', 'e3', 'e3', 'e3', 'e3', 'e3', None, None, 'e4',
        'e4', 'e4', 'e4', 'e4', 'e4', 'e4', None, None]))

    assert_allclose(tt.s, np.array([
        0.        , 1.        , 2.        , 2.        , 2.        ,
        2.16666667, 2.16666667, 2.83333333, 2.83333333, 3.        ,
        3.        , 3.        , 3.        , 3.        , 3.16666667,
        3.16666667, 3.83333333, 3.83333333, 4.        , 4.        ,
        4.        , 4.        , 4.        , 4.16666667, 4.16666667,
        4.83333333, 4.83333333, 5.        , 5.        , 5.        ]))

    assert_allclose(tt_no_repl.s, np.array([
        0.        , 1.        , 2.        , 2.        , 2.        ,
        2.16666667, 2.16666667, 2.83333333, 2.83333333, 3.        ,
        3.        , 3.        , 3.        , 3.        , 3.16666667,
        3.16666667, 3.83333333, 3.83333333, 4.        , 4.        ,
        4.        , 4.        , 4.        , 4.16666667, 4.16666667,
        4.83333333, 4.83333333, 5.        , 5.        , 5.        ]))


    p1 = p0.copy()
    p2 = p0.copy()

    line.track(p1)
    line_no_repl.track(p2)

    assert_allclose(p2.x, p1.x, rtol=0, atol=1e-14)
    assert_allclose(p2.px, p1.px, rtol=0, atol=1e-14)
    assert_allclose(p2.y, p1.y, rtol=0, atol=1e-14)
    assert_allclose(p2.py, p1.py, rtol=0, atol=1e-14)
    assert_allclose(p2.zeta, p1.zeta, rtol=0, atol=1e-14)
    assert_allclose(p2.delta, p1.delta, rtol=0, atol=1e-14)

    line.discard_tracker()
    line.insert_element(name='mkins1', element=xt.Marker(), at_s=0.5)
    line.insert_element(name='mkins2', element=xt.Marker(), at_s=1.5)
    line.insert_element(name='mkins3', element=xt.Marker(), at_s=2.5)
    line.insert_element(name='mkins4', element=xt.Marker(), at_s=3.5)
    line.insert_element(name='mkins5', element=xt.Marker(), at_s=4.5)
    line.build_tracker(_context=test_context)

    line_no_repl.discard_tracker()
    line_no_repl.insert_element(name='mkins1', element=xt.Marker(), at_s=0.5)
    line_no_repl.insert_element(name='mkins2', element=xt.Marker(), at_s=1.5)
    line_no_repl.insert_element(name='mkins3', element=xt.Marker(), at_s=2.5)
    line_no_repl.insert_element(name='mkins4', element=xt.Marker(), at_s=3.5)
    line_no_repl.insert_element(name='mkins5', element=xt.Marker(), at_s=4.5)
    line_no_repl.build_tracker(_context=test_context)

    tt = line.get_table()
    tt_no_repl = line_no_repl.get_table()


    assert np.all(tt.name == np.array(['e0_entry', 'e0..entry_map', 'e0..0', 'mkins1', 'e0..1',
        'e0..exit_map', 'e0_exit', 'e1_entry', 'e1..entry_map', 'e1..0',
        'mkins2', 'e1..1', 'e1..exit_map', 'e1_exit', 'e2_entry',
        'e2..entry_map', 'drift_e2..0', 'e2..0', 'drift_e2..1..0',
        'mkins3', 'drift_e2..1..1', 'e2..1', 'drift_e2..2', 'e2..exit_map',
        'e2_exit', 'e3_entry', 'e3..entry_map', 'drift_e3..0', 'e3..0',
        'drift_e3..1..0', 'mkins4', 'drift_e3..1..1', 'e3..1',
        'drift_e3..2', 'e3..exit_map', 'e3_exit', 'e4_entry',
        'e4..entry_map', 'drift_e4..0', 'e4..0', 'drift_e4..1..0',
        'mkins5', 'drift_e4..1..1', 'e4..1', 'drift_e4..2', 'e4..exit_map',
        'e4_exit', '_end_point']))

    assert np.all(tt_no_repl.name == np.array(['e0_entry', 'e0..entry_map', 'e0..0', 'mkins1', 'e0..1',
        'e0..exit_map', 'e0_exit', 'e1_entry', 'e1..entry_map', 'e1..0',
        'mkins2', 'e1..1', 'e1..exit_map', 'e1_exit', 'e2_entry',
        'e2..entry_map', 'drift_e2..0', 'e2..0', 'drift_e2..1..0',
        'mkins3', 'drift_e2..1..1', 'e2..1', 'drift_e2..2', 'e2..exit_map',
        'e2_exit', 'e3_entry', 'e3..entry_map', 'drift_e3..0', 'e3..0',
        'drift_e3..1..0', 'mkins4', 'drift_e3..1..1', 'e3..1',
        'drift_e3..2', 'e3..exit_map', 'e3_exit', 'e4_entry',
        'e4..entry_map', 'drift_e4..0', 'e4..0', 'drift_e4..1..0',
        'mkins5', 'drift_e4..1..1', 'e4..1', 'drift_e4..2', 'e4..exit_map',
        'e4_exit', '_end_point']))

    assert np.all(tt.parent_name == np.array([
        None, 'e0', 'e0', None, 'e0', 'e0', None, None, 'e0', 'e0', None,
        'e0', 'e0', None, None, 'e2', 'e2', 'e2', 'e2', None, 'e2', 'e2',
        'e2', 'e2', None, None, 'e2', 'e2', 'e2', 'e2', None, 'e2', 'e2',
        'e2', 'e2', None, None, 'e2', 'e2', 'e2', 'e2', None, 'e2', 'e2',
        'e2', 'e2', None, None]))

    assert np.all(tt_no_repl.parent_name == np.array([
        None, 'e0', 'e0', None, 'e0', 'e0', None, None, 'e1', 'e1', None,
        'e1', 'e1', None, None, 'e2', 'e2', 'e2', 'e2', None, 'e2', 'e2',
        'e2', 'e2', None, None, 'e3', 'e3', 'e3', 'e3', None, 'e3', 'e3',
        'e3', 'e3', None, None, 'e4', 'e4', 'e4', 'e4', None, 'e4', 'e4',
        'e4', 'e4', None, None]))

    assert_allclose(tt.s, [
        0.        , 0.        , 0.        , 0.5       , 0.5       ,
        1.        , 1.        , 1.        , 1.        , 1.        ,
        1.5       , 1.5       , 2.        , 2.        , 2.        ,
        2.        , 2.        , 2.16666667, 2.16666667, 2.5       ,
        2.5       , 2.83333333, 2.83333333, 3.        , 3.        ,
        3.        , 3.        , 3.        , 3.16666667, 3.16666667,
        3.5       , 3.5       , 3.83333333, 3.83333333, 4.        ,
        4.        , 4.        , 4.        , 4.        , 4.16666667,
        4.16666667, 4.5       , 4.5       , 4.83333333, 4.83333333,
        5.        , 5.        , 5.        ], rtol=0, atol=5e-9)

    assert_allclose(tt_no_repl.s, [
        0.        , 0.        , 0.        , 0.5       , 0.5       ,
        1.        , 1.        , 1.        , 1.        , 1.        ,
        1.5       , 1.5       , 2.        , 2.        , 2.        ,
        2.        , 2.        , 2.16666667, 2.16666667, 2.5       ,
        2.5       , 2.83333333, 2.83333333, 3.        , 3.        ,
        3.        , 3.        , 3.        , 3.16666667, 3.16666667,
        3.5       , 3.5       , 3.83333333, 3.83333333, 4.        ,
        4.        , 4.        , 4.        , 4.        , 4.16666667,
        4.16666667, 4.5       , 4.5       , 4.83333333, 4.83333333,
        5.        , 5.        , 5.        ], rtol=0, atol=5e-9)

    p1 = p0.copy()
    p2 = p0.copy()

    line.track(p1)
    line_no_repl.track(p2)

    assert_allclose(p2.x, p1.x, rtol=0, atol=1e-14)
    assert_allclose(p2.px, p1.px, rtol=0, atol=1e-14)
    assert_allclose(p2.y, p1.y, rtol=0, atol=1e-14)
    assert_allclose(p2.py, p1.py, rtol=0, atol=1e-14)
    assert_allclose(p2.zeta, p1.zeta, rtol=0, atol=1e-14)
    assert_allclose(p2.delta, p1.delta, rtol=0, atol=1e-14)

@for_all_test_contexts
def test_slice_thick_and_insert_with_replicas(test_context):

    elements = {
        'e0': xt.Bend(k0=0.3, h=0.31, length=1),
        'e1': xt.Replica(parent_name='e0'),
        'e2': xt.Bend(k0=-0.4, h=-0.41, length=1),
        'e3': xt.Replica(parent_name='e2'),
        'e4': xt.Replica(parent_name='e3'), # Replica of replica
    }
    line = xt.Line(elements=elements,
                element_names=list(elements.keys()))
    line.build_tracker(_context=test_context)

    assert line['e2']._movable

    element_no_repl={
        'e0': xt.Bend(k0=0.3, h=0.31, length=1),
        'e1': xt.Bend(k0=0.3, h=0.31, length=1),
        'e2': xt.Bend(k0=-0.4, h=-0.41, length=1),
        'e3': xt.Bend(k0=-0.4, h=-0.41, length=1),
        'e4': xt.Bend(k0=-0.4, h=-0.41, length=1),
    }

    line_no_repl = xt.Line(elements=element_no_repl,
                        element_names=list(element_no_repl.keys()))
    line_no_repl.build_tracker(_context=test_context)

    tt = line.get_table()
    tt_no_repl = line_no_repl.get_table()

    assert np.all(tt.name == ['e0', 'e1', 'e2', 'e3', 'e4', '_end_point'])
    assert np.all(tt_no_repl.name == ['e0', 'e1', 'e2', 'e3', 'e4', '_end_point'])

    assert np.all(tt.parent_name == [None, 'e0', None, 'e2', 'e3', None])
    assert np.all(tt_no_repl.parent_name == [None, None, None, None, None, None])

    assert_allclose(tt.s, [0., 1., 2., 3., 4., 5.], rtol=0, atol=1e-14)
    assert_allclose(tt_no_repl.s, [0., 1., 2., 3., 4., 5.], rtol=0, atol=1e-14)

    p0 = xt.Particles(p0c=1e9, x=0.1, px=0.2, y=0.3, py=0.4, delta=0.03,
                        _context=test_context)
    p1 = p0.copy()
    p2 = p0.copy()

    line.track(p1)
    line_no_repl.track(p2)

    assert_allclose(p2.x, p1.x, rtol=0, atol=1e-14)
    assert_allclose(p2.px, p1.px, rtol=0, atol=1e-14)
    assert_allclose(p2.y, p1.y, rtol=0, atol=1e-14)
    assert_allclose(p2.py, p1.py, rtol=0, atol=1e-14)
    assert_allclose(p2.zeta, p1.zeta, rtol=0, atol=1e-14)
    assert_allclose(p2.delta, p1.delta, rtol=0, atol=1e-14)

    assert line['e2']._movable
    # line['e2']._mark = True

    line.slice_thick_elements(
        slicing_strategies=[
            xt.Strategy(None),
            xt.Strategy(xt.Teapot(3, mode='thick'), name='e2|e3|e4')])
    assert line['e2']._movable
    line.build_tracker(_context=test_context)

    assert line['e2']._movable

    line_no_repl.slice_thick_elements(
        slicing_strategies=[
            xt.Strategy(None),
            xt.Strategy(xt.Teapot(3, mode='thick'), name='e2|e3|e4')])
    line_no_repl.build_tracker(_context=test_context)

    assert line['e2']._movable

    tt = line.get_table()
    tt_no_repl = line_no_repl.get_table()

    assert np.all(tt.name == np.array([
        'e0', 'e1', 'e2_entry', 'e2..entry_map', 'e2..0', 'e2..1', 'e2..2',
        'e2..exit_map', 'e2_exit', 'e3_entry', 'e3..entry_map', 'e3..0',
        'e3..1', 'e3..2', 'e3..exit_map', 'e3_exit', 'e4_entry',
        'e4..entry_map', 'e4..0', 'e4..1', 'e4..2', 'e4..exit_map',
        'e4_exit', '_end_point']))

    assert np.all(tt_no_repl.name == np.array([
        'e0', 'e1', 'e2_entry', 'e2..entry_map', 'e2..0', 'e2..1', 'e2..2',
        'e2..exit_map', 'e2_exit', 'e3_entry', 'e3..entry_map', 'e3..0',
        'e3..1', 'e3..2', 'e3..exit_map', 'e3_exit', 'e4_entry',
        'e4..entry_map', 'e4..0', 'e4..1', 'e4..2', 'e4..exit_map',
        'e4_exit', '_end_point']))

    assert np.all(tt.parent_name == np.array([
        None, 'e0', None, 'e2', 'e2', 'e2', 'e2', 'e2', None, None, 'e2',
        'e2', 'e2', 'e2', 'e2', None, None, 'e2', 'e2', 'e2', 'e2', 'e2',
        None, None]))
    assert np.all(tt_no_repl.parent_name == np.array([
        None, None, None, 'e2', 'e2', 'e2', 'e2', 'e2', None, None, 'e3',
        'e3', 'e3', 'e3', 'e3', None, None, 'e4', 'e4', 'e4', 'e4', 'e4',
        None, None]))

    assert_allclose(tt.s, np.array([
        0.        , 1.        , 2.        , 2.        , 2.        ,
        2.16666667, 2.83333333, 3.        , 3.        , 3.        ,
        3.        , 3.        , 3.16666667, 3.83333333, 4.        ,
        4.        , 4.        , 4.        , 4.        , 4.16666667,
        4.83333333, 5.        , 5.        , 5.        ]))

    assert_allclose(tt_no_repl.s, np.array([
        0.        , 1.        , 2.        , 2.        , 2.        ,
        2.16666667, 2.83333333, 3.        , 3.        , 3.        ,
        3.        , 3.        , 3.16666667, 3.83333333, 4.        ,
        4.        , 4.        , 4.        , 4.        , 4.16666667,
        4.83333333, 5.        , 5.        , 5.        ]))


    p1 = p0.copy()
    p2 = p0.copy()

    line.track(p1)
    line_no_repl.track(p2)

    assert_allclose(p2.x, p1.x, rtol=0, atol=1e-14)
    assert_allclose(p2.px, p1.px, rtol=0, atol=1e-14)
    assert_allclose(p2.y, p1.y, rtol=0, atol=1e-14)
    assert_allclose(p2.py, p1.py, rtol=0, atol=1e-14)
    assert_allclose(p2.zeta, p1.zeta, rtol=0, atol=1e-14)
    assert_allclose(p2.delta, p1.delta, rtol=0, atol=1e-14)

    assert line['e2']._movable

    line.discard_tracker()
    assert line['e2']._movable
    line.insert_element(name='mkins1', element=xt.Marker(), at_s=0.5)
    line.insert_element(name='mkins2', element=xt.Marker(), at_s=1.5)
    line.insert_element(name='mkins3', element=xt.Marker(), at_s=2.5)
    line.insert_element(name='mkins4', element=xt.Marker(), at_s=3.5)
    line.insert_element(name='mkins5', element=xt.Marker(), at_s=4.5)
    line.build_tracker(_context=test_context)
    assert line['e2']._movable

    line_no_repl.discard_tracker()
    line_no_repl.insert_element(name='mkins1', element=xt.Marker(), at_s=0.5)
    line_no_repl.insert_element(name='mkins2', element=xt.Marker(), at_s=1.5)
    line_no_repl.insert_element(name='mkins3', element=xt.Marker(), at_s=2.5)
    line_no_repl.insert_element(name='mkins4', element=xt.Marker(), at_s=3.5)
    line_no_repl.insert_element(name='mkins5', element=xt.Marker(), at_s=4.5)
    line_no_repl.build_tracker(_context=test_context)
    assert line['e2']._movable

    tt = line.get_table()
    tt_no_repl = line_no_repl.get_table()


    assert np.all(tt.name == np.array([
        'e0_entry', 'e0..entry_map', 'e0..0', 'mkins1', 'e0..1',
        'e0..exit_map', 'e0_exit', 'e1_entry', 'e1..entry_map', 'e1..0',
        'mkins2', 'e1..1', 'e1..exit_map', 'e1_exit', 'e2_entry',
        'e2..entry_map', 'e2..0', 'e2..1_entry', 'e2..1..0', 'mkins3',
        'e2..1..1', 'e2..1_exit', 'e2..2', 'e2..exit_map', 'e2_exit',
        'e3_entry', 'e3..entry_map', 'e3..0', 'e3..1_entry', 'e3..1..0',
        'mkins4', 'e3..1..1', 'e3..1_exit', 'e3..2', 'e3..exit_map',
        'e3_exit', 'e4_entry', 'e4..entry_map', 'e4..0', 'e4..1_entry',
        'e4..1..0', 'mkins5', 'e4..1..1', 'e4..1_exit', 'e4..2',
        'e4..exit_map', 'e4_exit', '_end_point']))

    assert np.all(tt_no_repl.name == np.array([
        'e0_entry', 'e0..entry_map', 'e0..0', 'mkins1', 'e0..1',
        'e0..exit_map', 'e0_exit', 'e1_entry', 'e1..entry_map', 'e1..0',
        'mkins2', 'e1..1', 'e1..exit_map', 'e1_exit', 'e2_entry',
        'e2..entry_map', 'e2..0', 'e2..1_entry', 'e2..1..0', 'mkins3',
        'e2..1..1', 'e2..1_exit', 'e2..2', 'e2..exit_map', 'e2_exit',
        'e3_entry', 'e3..entry_map', 'e3..0', 'e3..1_entry', 'e3..1..0',
        'mkins4', 'e3..1..1', 'e3..1_exit', 'e3..2', 'e3..exit_map',
        'e3_exit', 'e4_entry', 'e4..entry_map', 'e4..0', 'e4..1_entry',
        'e4..1..0', 'mkins5', 'e4..1..1', 'e4..1_exit', 'e4..2',
        'e4..exit_map', 'e4_exit', '_end_point']))

    assert np.all(tt.parent_name == np.array([
        None, 'e0', 'e0', None, 'e0', 'e0', None, None, 'e0', 'e0', None,
        'e0', 'e0', None, None, 'e2', 'e2', None, 'e2', None, 'e2', None,
        'e2', 'e2', None, None, 'e2', 'e2', None, 'e2', None, 'e2', None,
        'e2', 'e2', None, None, 'e2', 'e2', None, 'e2', None, 'e2', None,
        'e2', 'e2', None, None]))

    assert np.all(tt_no_repl.parent_name == np.array([
        None, 'e0', 'e0', None, 'e0', 'e0', None, None, 'e1', 'e1', None,
        'e1', 'e1', None, None, 'e2', 'e2', None, 'e2', None, 'e2', None,
        'e2', 'e2', None, None, 'e3', 'e3', None, 'e3', None, 'e3', None,
        'e3', 'e3', None, None, 'e4', 'e4', None, 'e4', None, 'e4', None,
        'e4', 'e4', None, None]))

    assert_allclose(tt.s, [
        0.        , 0.        , 0.        , 0.5       , 0.5       ,
        1.        , 1.        , 1.        , 1.        , 1.        ,
        1.5       , 1.5       , 2.        , 2.        , 2.        ,
        2.        , 2.        , 2.16666667, 2.16666667, 2.5       ,
        2.5       , 2.83333333, 2.83333333, 3.        , 3.        ,
        3.        , 3.        , 3.        , 3.16666667, 3.16666667,
        3.5       , 3.5       , 3.83333333, 3.83333333, 4.        ,
        4.        , 4.        , 4.        , 4.        , 4.16666667,
        4.16666667, 4.5       , 4.5       , 4.83333333, 4.83333333,
        5.        , 5.        , 5.        ], rtol=0, atol=5e-9)

    assert_allclose(tt_no_repl.s, [
        0.        , 0.        , 0.        , 0.5       , 0.5       ,
        1.        , 1.        , 1.        , 1.        , 1.        ,
        1.5       , 1.5       , 2.        , 2.        , 2.        ,
        2.        , 2.        , 2.16666667, 2.16666667, 2.5       ,
        2.5       , 2.83333333, 2.83333333, 3.        , 3.        ,
        3.        , 3.        , 3.        , 3.16666667, 3.16666667,
        3.5       , 3.5       , 3.83333333, 3.83333333, 4.        ,
        4.        , 4.        , 4.        , 4.        , 4.16666667,
        4.16666667, 4.5       , 4.5       , 4.83333333, 4.83333333,
        5.        , 5.        , 5.        ], rtol=0, atol=5e-9)

    p1 = p0.copy()
    p2 = p0.copy()

    line.track(p1)
    line_no_repl.track(p2)
    assert line['e2']._movable

    assert_allclose(p2.x, p1.x, rtol=0, atol=1e-14)
    assert_allclose(p2.px, p1.px, rtol=0, atol=1e-14)
    assert_allclose(p2.y, p1.y, rtol=0, atol=1e-14)
    assert_allclose(p2.py, p1.py, rtol=0, atol=1e-14)
    assert_allclose(p2.zeta, p1.zeta, rtol=0, atol=1e-14)
    assert_allclose(p2.delta, p1.delta, rtol=0, atol=1e-14)
