import xtrack as xt
import numpy as np
import xobjects as xo
from xobjects.test_helpers import for_all_test_contexts

@for_all_test_contexts
def test_rbend_straight_body_edge_full(test_context):

    edge_model = 'full'

    b_ref = xt.RBend(angle=0.1, k0_from_h=True, length_straight=3.)
    b_ref.edge_entry_model = edge_model
    b_ref.edge_exit_model = edge_model
    b_ref.model = 'rot-kick-rot'
    b_ref.num_multipole_kicks = 100
    l_ref = xt.Line([b_ref])
    l_ref.append('end', xt.Marker())
    l_ref.particle_ref = xt.Particles(p0c=10e9)
    l_ref.build_tracker(_context=test_context)
    tw_ref0 = l_ref.twiss(betx=1, bety=1)
    tw_ref = l_ref.twiss(betx=1, bety=1, x=2e-3, px=1e-3, y=2e-3, py=2e-3, delta=1e-3)

    b_test = xt.RBend(
        angle=0.1, k0_from_h=True, length_straight=3)
    b_test.rbend_model = 'straight-body'
    b_test.model = 'bend-kick-bend'
    b_test.num_multipole_kicks = 100
    b_test.edge_entry_model = edge_model
    b_test.edge_exit_model = edge_model
    l_test = xt.Line([b_test])
    l_test.append('end', xt.Marker())
    l_test.particle_ref = xt.Particles(p0c=10e9)
    l_test.build_tracker(_context=test_context)
    tw_test0 = l_test.twiss(betx=1, bety=1)
    tw_test = l_test.twiss(betx=1, bety=1, x=2e-3, px=1e-3, y=2e-3, py=2e-3, delta=1e-3)

    xo.assert_allclose(tw_ref0.betx, tw_test0.betx, rtol=1e-9, atol=0.0)
    xo.assert_allclose(tw_ref0.bety, tw_test0.bety, rtol=1e-9, atol=0.0)
    xo.assert_allclose(tw_ref0.x,    tw_test0.x, rtol=0, atol=1e-12)
    xo.assert_allclose(tw_ref0.y,    tw_test0.y, rtol=0, atol=1e-12)
    xo.assert_allclose(tw_ref0.s,    tw_test0.s, rtol=0, atol=1e-12)
    xo.assert_allclose(tw_ref0.zeta, tw_test0.zeta, rtol=0, atol=1e-11)
    xo.assert_allclose(tw_ref0.px,   tw_test0.px, rtol=0, atol=1e-12)
    xo.assert_allclose(tw_ref0.py,   tw_test0.py, rtol=0, atol=1e-12)

    xo.assert_allclose(tw_ref.betx, tw_test.betx, rtol=5e-9, atol=0.0)
    xo.assert_allclose(tw_ref.bety, tw_test.bety, rtol=5e-9, atol=0.0)
    xo.assert_allclose(tw_ref.x,    tw_test.x, rtol=0, atol=1e-12)
    xo.assert_allclose(tw_ref.y,    tw_test.y, rtol=0, atol=1e-12)
    xo.assert_allclose(tw_ref.s,    tw_test.s, rtol=0, atol=1e-12)
    xo.assert_allclose(tw_ref.zeta, tw_test.zeta, rtol=0, atol=1e-1)
    xo.assert_allclose(tw_ref.px,   tw_test.px, rtol=0, atol=1e-12)
    xo.assert_allclose(tw_ref.py,   tw_test.py, rtol=0, atol=1e-12)

    l_sliced = l_test.copy(shallow=True)
    l_sliced.cut_at_s(np.linspace(0, l_test.get_length(), 100))
    tw_test_sliced0 = l_sliced.twiss(betx=1, bety=1)

    xo.assert_allclose(tw_test_sliced0.betx[-1], tw_test0.betx[-1], rtol=1e-9, atol=0.0)
    xo.assert_allclose(tw_test_sliced0.bety[-1], tw_test0.bety[-1], rtol=1e-9, atol=0.0)
    xo.assert_allclose(tw_test_sliced0.x[-1],    tw_test0.x[-1], rtol=0, atol=1e-12)
    xo.assert_allclose(tw_test_sliced0.y[-1],    tw_test0.y[-1], rtol=0, atol=1e-12)
    xo.assert_allclose(tw_test_sliced0.s[-1],    tw_test0.s[-1], rtol=0, atol=1e-12)
    xo.assert_allclose(tw_test_sliced0.zeta[-1], tw_test0.zeta[-1], rtol=0, atol=1e-11)
    xo.assert_allclose(tw_test_sliced0.px[-1],   tw_test0.px[-1], rtol=0, atol=1e-12)
    xo.assert_allclose(tw_test_sliced0.py[-1],   tw_test0.py[-1], rtol=0, atol=1e-12)

@for_all_test_contexts
def test_rbend_straight_body_edge_linear(test_context):

    edge_model = 'linear'

    b_ref = xt.RBend(angle=0.1, k0_from_h=True, length_straight=3.)
    b_ref.edge_entry_model = edge_model
    b_ref.edge_exit_model = edge_model
    b_ref.model = 'rot-kick-rot'
    b_ref.num_multipole_kicks = 100
    l_ref = xt.Line([b_ref])
    l_ref.append('end', xt.Marker())
    l_ref.particle_ref = xt.Particles(p0c=10e9)
    l_ref.build_tracker(_context=test_context)
    tw_ref0 = l_ref.twiss(betx=1, bety=1)
    tw_ref = l_ref.twiss(betx=1, bety=1, x=2e-3, px=1e-3, y=2e-3, py=2e-3, delta=1e-3)

    b_test = xt.RBend(
        angle=0.1, k0_from_h=True, length_straight=3)
    b_test.rbend_model = 'straight-body'
    b_test.model = 'bend-kick-bend'
    b_test.num_multipole_kicks = 100
    b_test.edge_entry_model = edge_model
    b_test.edge_exit_model = edge_model
    l_test = xt.Line([b_test])
    l_test.append('end', xt.Marker())
    l_test.particle_ref = xt.Particles(p0c=10e9)
    l_test.build_tracker(_context=test_context)
    tw_test0 = l_test.twiss(betx=1, bety=1)
    tw_test = l_test.twiss(betx=1, bety=1, x=2e-3, px=1e-3, y=2e-3, py=2e-3, delta=1e-3)

    xo.assert_allclose(tw_ref0.betx, tw_test0.betx, rtol=1e-9, atol=0.0)
    xo.assert_allclose(tw_ref0.bety, tw_test0.bety, rtol=1e-9, atol=0.0)
    xo.assert_allclose(tw_ref0.x,    tw_test0.x, rtol=0, atol=1e-12)
    xo.assert_allclose(tw_ref0.y,    tw_test0.y, rtol=0, atol=1e-12)
    xo.assert_allclose(tw_ref0.s,    tw_test0.s, rtol=0, atol=1e-12)
    xo.assert_allclose(tw_ref0.zeta, tw_test0.zeta, rtol=0, atol=1e-11)
    xo.assert_allclose(tw_ref0.px,   tw_test0.px, rtol=0, atol=1e-12)
    xo.assert_allclose(tw_ref0.py,   tw_test0.py, rtol=0, atol=1e-12)

    xo.assert_allclose(tw_ref.betx, tw_test.betx, rtol=5e-6, atol=0.0)
    xo.assert_allclose(tw_ref.bety, tw_test.bety, rtol=5e-6, atol=0.0)
    xo.assert_allclose(tw_ref.x,    tw_test.x, rtol=0, atol=1e-8)
    xo.assert_allclose(tw_ref.y,    tw_test.y, rtol=0, atol=1e-8)
    xo.assert_allclose(tw_ref.s,    tw_test.s, rtol=0, atol=1e-12)
    xo.assert_allclose(tw_ref.zeta, tw_test.zeta, rtol=0, atol=1e-1)
    xo.assert_allclose(tw_ref.px,   tw_test.px, rtol=0, atol=1e-9)
    xo.assert_allclose(tw_ref.py,   tw_test.py, rtol=0, atol=1e-9)

    tw_back = l_test.twiss(init=tw_test, init_at='end')

    assert tw_back.orientation == 'backward'
    xo.assert_allclose(tw_back.betx, tw_test.betx, rtol=5e-6, atol=0.0)
    xo.assert_allclose(tw_back.bety, tw_test.bety, rtol=5e-6, atol=0.0)
    xo.assert_allclose(tw_back.x,    tw_test.x, rtol=0, atol=1e-8)
    xo.assert_allclose(tw_back.y,    tw_test.y, rtol=0, atol=1e-8)
    xo.assert_allclose(tw_back.s,    tw_test.s, rtol=0, atol=1e-12)
    xo.assert_allclose(tw_back.zeta, tw_test.zeta, rtol=0, atol=1e-1)
    xo.assert_allclose(tw_back.px,   tw_test.px, rtol=0, atol=1e-9)
    xo.assert_allclose(tw_back.py,   tw_test.py, rtol=0, atol=1e-9)

    l_sliced = l_test.copy(shallow=True)
    l_sliced.cut_at_s(np.linspace(0, l_test.get_length(), 100))
    tw_test_sliced0 = l_sliced.twiss(betx=1, bety=1)

    xo.assert_allclose(tw_test_sliced0.betx[-1], tw_test0.betx[-1], rtol=1e-9, atol=0.0)
    xo.assert_allclose(tw_test_sliced0.bety[-1], tw_test0.bety[-1], rtol=1e-9, atol=0.0)
    xo.assert_allclose(tw_test_sliced0.x[-1],    tw_test0.x[-1], rtol=0, atol=1e-12)
    xo.assert_allclose(tw_test_sliced0.y[-1],    tw_test0.y[-1], rtol=0, atol=1e-12)
    xo.assert_allclose(tw_test_sliced0.s[-1],    tw_test0.s[-1], rtol=0, atol=1e-12)
    xo.assert_allclose(tw_test_sliced0.zeta[-1], tw_test0.zeta[-1], rtol=0, atol=1e-11)
    xo.assert_allclose(tw_test_sliced0.px[-1],   tw_test0.px[-1], rtol=0, atol=1e-12)
    xo.assert_allclose(tw_test_sliced0.py[-1],   tw_test0.py[-1], rtol=0, atol=1e-12)

    tw_test_sliced_back = l_sliced.twiss(init=tw_test_sliced0, init_at='end')

    assert tw_test_sliced_back.orientation == 'backward'
    xo.assert_allclose(tw_test_sliced_back.betx, tw_test_sliced0.betx, rtol=5e-9, atol=0.0)
    xo.assert_allclose(tw_test_sliced_back.bety, tw_test_sliced0.bety, rtol=5e-9, atol=0.0)
    xo.assert_allclose(tw_test_sliced_back.x,    tw_test_sliced0.x, rtol=0, atol=1e-12)
    xo.assert_allclose(tw_test_sliced_back.y,    tw_test_sliced0.y, rtol=0, atol=1e-12)
    xo.assert_allclose(tw_test_sliced_back.s,    tw_test_sliced0.s, rtol=0, atol=1e-12)
    xo.assert_allclose(tw_test_sliced_back.zeta, tw_test_sliced0.zeta, rtol=0, atol=1e-11)
    xo.assert_allclose(tw_test_sliced_back.px,   tw_test_sliced0.px, rtol=0, atol=1e-12)
    xo.assert_allclose(tw_test_sliced_back.py,   tw_test_sliced0.py, rtol=0, atol=1e-12)

def test_rbend_straight_body_survey():

    env = xt.Environment(particle_ref=xt.Particles(p0c=10e9))

    line = env.new_line(length=5, components=[
        env.new('mb', 'RBend', angle=0.3, k0_from_h=True, length_straight=3,
                rbend_model='straight-body', at=2.5)])

    line.cut_at_s(np.linspace(0, line.get_length(), 11))
    line.insert('mid', xt.Marker(), at=2.5)

    line['mb'].rbend_model = 'straight-body'
    sv_straight = line.survey(element0='mid')
    tt_straight = line.get_table(attr=True)

    line['mb'].rbend_model = 'curved-body'
    sv_curved = line.survey(element0='mid')
    tt_curved = line.get_table(attr=True)

    tt_straight.cols['s element_type angle_rad']
    # is:
    # Table: 18 rows, 4 cols
    # name                      s element_type            angle_rad
    # drift_1..0                0 DriftSlice                      0
    # drift_1..1              0.5 DriftSlice                      0
    # mb_entry            0.99436 Marker                          0
    # mb..entry_map       0.99436 ThinSliceRBendEntry          0.15
    # mb..0               0.99436 ThickSliceRBend                 0
    # mb..1                     1 ThickSliceRBend                 0
    # mb..2                   1.5 ThickSliceRBend                 0
    # mb..3                     2 ThickSliceRBend                 0
    # mid                     2.5 Marker                          0
    # mb..4                   2.5 ThickSliceRBend                 0
    # mb..5                     3 ThickSliceRBend                 0
    # mb..6                   3.5 ThickSliceRBend                 0
    # mb..7                     4 ThickSliceRBend                 0
    # mb..exit_map        4.00564 ThinSliceRBendExit           0.15
    # mb_exit             4.00564 Marker                          0
    # drift_2..0          4.00564 DriftSlice                      0
    # drift_2..1              4.5 DriftSlice                      0
    # _end_point                5                                 0

    assert np.all(tt_straight['name'] == [
        'drift_1..0', 'drift_1..1', 'mb_entry', 'mb..entry_map', 'mb..0',
        'mb..1', 'mb..2', 'mb..3', 'mid', 'mb..4', 'mb..5', 'mb..6', 'mb..7',
        'mb..exit_map', 'mb_exit', 'drift_2..0', 'drift_2..1', '_end_point'
    ])

    # Assert entire columns using np.all
    assert np.all(tt_straight['element_type'] == [
        'DriftSlice', 'DriftSlice', 'Marker', 'ThinSliceRBendEntry', 'ThickSliceRBend',
        'ThickSliceRBend', 'ThickSliceRBend', 'ThickSliceRBend', 'Marker', 'ThickSliceRBend',
        'ThickSliceRBend', 'ThickSliceRBend', 'ThickSliceRBend', 'ThinSliceRBendExit',
        'Marker', 'DriftSlice', 'DriftSlice', ''
    ])

    xo.assert_allclose(
        tt_straight['angle_rad'],
        np.array([0, 0, 0, 0.15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.15, 0, 0, 0, 0]),
        atol=1e-12
    )

    xo.assert_allclose(tt_straight['s'], np.array([
        0.       , 0.5      , 0.9943602, 0.9943602, 0.9943602, 1.       ,
        1.5      , 2.       , 2.5      , 2.5      , 3.       , 3.5      ,
        4.       , 4.0056398, 4.0056398, 4.0056398, 4.5      , 5.       ]
    ), atol=1e-5)


    tt_curved.cols['s element_type angle_rad']
    # is:
    # Table: 18 rows, 4 cols
    # name                      s element_type            angle_rad
    # drift_1..0                0 DriftSlice                      0
    # drift_1..1              0.5 DriftSlice                      0
    # mb_entry            0.99436 Marker                          0
    # mb..entry_map       0.99436 ThinSliceRBendEntry             0
    # mb..0               0.99436 ThickSliceRBend       0.000561868
    # mb..1                     1 ThickSliceRBend         0.0498127
    # mb..2                   1.5 ThickSliceRBend         0.0498127
    # mb..3                     2 ThickSliceRBend         0.0498127
    # mid                     2.5 Marker                          0
    # mb..4                   2.5 ThickSliceRBend         0.0498127
    # mb..5                     3 ThickSliceRBend         0.0498127
    # mb..6                   3.5 ThickSliceRBend         0.0498127
    # mb..7                     4 ThickSliceRBend       0.000561868
    # mb..exit_map        4.00564 ThinSliceRBendExit              0
    # mb_exit             4.00564 Marker                          0
    # drift_2..0          4.00564 DriftSlice                      0
    # drift_2..1              4.5 DriftSlice                      0
    # _end_point                5                                 0

    assert np.all(tt_curved['name'] == [
        'drift_1..0', 'drift_1..1', 'mb_entry', 'mb..entry_map', 'mb..0',
        'mb..1', 'mb..2', 'mb..3', 'mid', 'mb..4', 'mb..5', 'mb..6', 'mb..7',
        'mb..exit_map', 'mb_exit', 'drift_2..0', 'drift_2..1', '_end_point'
    ])

    assert np.all(tt_curved['element_type'] == [
        'DriftSlice', 'DriftSlice', 'Marker', 'ThinSliceRBendEntry', 'ThickSliceRBend',
        'ThickSliceRBend', 'ThickSliceRBend', 'ThickSliceRBend', 'Marker', 'ThickSliceRBend',
        'ThickSliceRBend', 'ThickSliceRBend', 'ThickSliceRBend', 'ThinSliceRBendExit',
        'Marker', 'DriftSlice', 'DriftSlice', ''
    ])

    xo.assert_allclose(
        tt_curved['angle_rad'],
        np.array([0, 0, 0, 0, 0.000561868, 0.0498127, 0.0498127, 0.0498127, 0, 0.0498127,
                0.0498127, 0.0498127, 0.000561868, 0, 0, 0, 0, 0]),
        rtol=1e-6
    )

    xo.assert_allclose(tt_curved['s'], np.array([
        0.       , 0.5      , 0.9943602, 0.9943602, 0.9943602, 1.       ,
        1.5      , 2.       , 2.5      , 2.5      , 3.       , 3.5      ,
        4.       , 4.0056398, 4.0056398, 4.0056398, 4.5      , 5.       ]
    ), atol=1e-5)