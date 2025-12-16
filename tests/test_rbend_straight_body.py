import xtrack as xt
import numpy as np
import xobjects as xo
from cpymad.madx import Madx
from xobjects.test_helpers import for_all_test_contexts
import pytest

import pathlib

test_data_folder = pathlib.Path(
        __file__).parent.joinpath('../test_data').absolute()

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

def test_rbend_straight_body_edge_full_angle_diff():

    edge_model = 'full'

    b_ref = xt.RBend(angle=0.1, k0_from_h=True, length_straight=3., rbend_angle_diff=0.1)
    b_ref.edge_entry_model = edge_model
    b_ref.edge_exit_model = edge_model
    b_ref.model = 'rot-kick-rot'
    b_ref.num_multipole_kicks = 100
    l_ref = xt.Line([b_ref])
    l_ref.append('end', xt.Marker())
    l_ref.particle_ref = xt.Particles(p0c=10e9)
    tw_ref0 = l_ref.twiss(betx=1, bety=1, strengths=True)
    tw_ref = l_ref.twiss(betx=1, bety=1, x=2e-3, px=1e-3, y=2e-3, py=2e-3, delta=1e-3)

    b_test = xt.RBend(
        angle=0.1, k0_from_h=True, length_straight=3, rbend_angle_diff=0.1,
        rbend_shift=0, rbend_compensate_sagitta=False)
    b_test.rbend_model = 'straight-body'
    b_test.model = 'bend-kick-bend'
    b_test.num_multipole_kicks = 100
    b_test.edge_entry_model = edge_model
    b_test.edge_exit_model = edge_model

    b_test.rbend_shift += b_test._x0_in

    l_test = xt.Line([b_test])
    l_test.append('end', xt.Marker())
    l_test.particle_ref = xt.Particles(p0c=10e9)
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

def test_rbend_straight_body_edge_linear_angle_diff():

    edge_model = 'linear'

    b_ref = xt.RBend(angle=0.1, k0_from_h=True, length_straight=3., rbend_angle_diff=0.1)
    b_ref.edge_entry_model = edge_model
    b_ref.edge_exit_model = edge_model
    b_ref.model = 'rot-kick-rot'
    b_ref.num_multipole_kicks = 100
    l_ref = xt.Line([b_ref])
    l_ref.append('end', xt.Marker())
    l_ref.particle_ref = xt.Particles(p0c=10e9)
    tw_ref0 = l_ref.twiss(betx=1, bety=1)
    tw_ref = l_ref.twiss(betx=1, bety=1, x=2e-3, px=1e-3, y=2e-3, py=2e-3, delta=1e-3)

    b_test = xt.RBend(
        angle=0.1, k0_from_h=True, length_straight=3, rbend_angle_diff=0.1,
        rbend_shift=0, rbend_compensate_sagitta=False)
    b_test.rbend_model = 'straight-body'
    b_test.model = 'bend-kick-bend'
    b_test.num_multipole_kicks = 100
    b_test.edge_entry_model = edge_model
    b_test.edge_exit_model = edge_model

    b_test.rbend_shift += b_test._x0_in

    l_test = xt.Line([b_test])
    l_test.append('end', xt.Marker())
    l_test.particle_ref = xt.Particles(p0c=10e9)
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
    xo.assert_allclose(tw_ref.x,    tw_test.x, rtol=0, atol=2e-8)
    xo.assert_allclose(tw_ref.y,    tw_test.y, rtol=0, atol=2e-8)
    xo.assert_allclose(tw_ref.s,    tw_test.s, rtol=0, atol=2e-12)
    xo.assert_allclose(tw_ref.zeta, tw_test.zeta, rtol=0, atol=1e-1)
    xo.assert_allclose(tw_ref.px,   tw_test.px, rtol=0, atol=5e-9)
    xo.assert_allclose(tw_ref.py,   tw_test.py, rtol=0, atol=5e-9)

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


def test_rbend_straight_sps():

    env = xt.load([test_data_folder / 'sps_thick/sps.seq',
                   test_data_folder / 'sps_thick/lhc_q20.str'])

    line = env['sps']
    line.particle_ref = xt.Particles(p0c=26e9, mass0=xt.PROTON_MASS_EV)

    tt = line.get_table()
    tt_rbend = tt.rows[tt.element_type == 'RBend']

    line.slice_thick_elements(
            slicing_strategies=[
                xt.Strategy(None),
                xt.Strategy(slicing=xt.Uniform(10, mode='thick'), element_type=xt.RBend),
        ])

    line.set(tt_rbend, edge_entry_model='full')
    line.set(tt_rbend, edge_exit_model='full')

    line.set(tt_rbend, rbend_model='straight-body')
    tw_straight = line.twiss4d()

    line.set(tt_rbend, rbend_model='curved-body')
    tw_curved = line.twiss4d()

    xo.assert_allclose(tw_curved.x.max(), 0, rtol=0, atol=1e-9)
    assert tw_straight.x.max() > 3e-3

    xo.assert_allclose(tw_straight.qx, tw_curved.qx, rtol=0, atol=1e-8)
    xo.assert_allclose(tw_straight.qy, tw_curved.qy, rtol=0, atol=1e-8)
    xo.assert_allclose(tw_straight.dqx, tw_curved.dqx, rtol=0, atol=1e-3)
    xo.assert_allclose(tw_straight.dqy, tw_curved.dqy, rtol=0, atol=1e-3)
    xo.assert_allclose(tw_straight.rows['qf.*|qd.*'].betx,
                    tw_curved.rows['qf.*|qd.*'].betx,
                    atol=0, rtol=1e-8)
    xo.assert_allclose(tw_straight.rows['qf.*|qd.*'].bety,
                    tw_curved.rows['qf.*|qd.*'].bety,
                    atol=0, rtol=1e-8)
    xo.assert_allclose(tw_straight.rows['qf.*|qd.*'].x, 0, atol=1e-10, rtol=0)
    xo.assert_allclose(tw_straight.rows['qf.*|qd.*'].y, 0, atol=1e-10, rtol=0)

    # Switch to electrons to check synchrotron radiation features

    # 20 GeV electrons (like in LEP times)
    env.particle_ref = xt.Particles(energy0=20e9, mass0=xt.ELECTRON_MASS_EV)
    line.particle_ref = env.particle_ref

    line['actcse.31632'].voltage = 4.2e+08
    line['actcse.31632'].frequency = 3e6
    line['actcse.31632'].lag = 180.

    line.configure_radiation(model='mean')

    line.set(tt_rbend, rbend_model='curved-body')
    tw_rad_curved = line.twiss(eneloss_and_damping=True,
                                radiation_integrals=True)

    line.set(tt_rbend, rbend_model='straight-body')
    tw_rad_straight = line.twiss(eneloss_and_damping=True,
                                radiation_integrals=True)

    xo.assert_allclose(tw_rad_straight.eq_gemitt_x,
                    tw_rad_curved.eq_gemitt_x, rtol=1e-2)
    xo.assert_allclose(tw_rad_straight.eq_gemitt_zeta,
                    tw_rad_curved.eq_gemitt_zeta, rtol=1e-2)
    xo.assert_allclose(tw_rad_straight.eq_gemitt_y,
                    tw_rad_curved.eq_gemitt_y, atol=1e-20)
    xo.assert_allclose(tw_rad_straight.rad_int_eq_gemitt_x,
                    tw_rad_curved.rad_int_eq_gemitt_x, rtol=1e-2)
    xo.assert_allclose(tw_rad_straight.rad_int_eq_gemitt_y,
                    tw_rad_curved.rad_int_eq_gemitt_y, atol=1e-20)

def test_rbend_straight_body_survey_h():

    env = xt.Environment(particle_ref=xt.Particles(p0c=10e9))

    edge_model = 'full'

    line = env.new_line(length=5, components=[
        env.new('mb', 'RBend', angle=0.3, k0_from_h=True, length_straight=3,
                model='bend-kick-bend',
                rbend_model='straight-body', edge_entry_model=edge_model, edge_exit_model=edge_model,
                at=2.5)])
    line.insert('start', xt.Marker(), at=0)
    line.append('end', xt.Marker())

    line_no_slice = line.copy(shallow=True)

    line.cut_at_s(np.linspace(0, line.get_length(), 11))
    line.insert('mid', xt.Marker(), at=2.5)

    line['mb'].rbend_model = 'straight-body'
    sv_straight = line.survey(element0='mid', X0=-line['mb'].sagitta/2)
    tt_straight = line.get_table(attr=True)
    tw_straight = line.twiss(betx=1, bety=1)
    p_straight = (sv_straight.p0 + tw_straight.x[:, None] * sv_straight['ex']
                                + tw_straight.y[:, None] * sv_straight['ey'])
    tw_straight['X'] = p_straight[:, 0]
    tw_straight['Y'] = p_straight[:, 1]
    tw_straight['Z'] = p_straight[:, 2]

    sv_straight_start = line.survey(element0='start',
                                    X0=sv_straight['X', 'start'],
                                    Y0=sv_straight['Y', 'start'],
                                    Z0=sv_straight['Z', 'start'],
                                    theta0=sv_straight['theta', 'start'],
                                    phi0=sv_straight['phi', 'start'],
                                    psi0=sv_straight['psi', 'start'])
    sv_straight_end = line.survey(element0='end',
                                    X0=sv_straight['X', 'end'],
                                    Y0=sv_straight['Y', 'end'],
                                    Z0=sv_straight['Z', 'end'],
                                    theta0=sv_straight['theta', 'end'],
                                    phi0=sv_straight['phi', 'end'],
                                    psi0=sv_straight['psi', 'end'])

    sv_no_slice_straight_start = line_no_slice.survey(element0='start',
                                    X0=sv_straight['X', 'start'],
                                    Y0=sv_straight['Y', 'start'],
                                    Z0=sv_straight['Z', 'start'],
                                    theta0=sv_straight['theta', 'start'],
                                    phi0=sv_straight['phi', 'start'],
                                    psi0=sv_straight['psi', 'start'])
    sv_no_slice_straight_end = line_no_slice.survey(element0='end',
                                    X0=sv_straight['X', 'end'],
                                    Y0=sv_straight['Y', 'end'],
                                    Z0=sv_straight['Z', 'end'],
                                    theta0=sv_straight['theta', 'end'],
                                    phi0=sv_straight['phi', 'end'],
                                    psi0=sv_straight['psi', 'end'])
    tw_no_slice_straight = line_no_slice.twiss(betx=1, bety=1)

    line['mb'].rbend_model = 'curved-body'
    sv_curved = line.survey(element0='mid')
    tt_curved = line.get_table(attr=True)
    tw_curved = line.twiss(betx=1, bety=1)
    p_curved = (sv_curved.p0 + tw_curved.x[:, None] * sv_curved['ex']
                            + tw_curved.y[:, None] * sv_curved['ey'])
    tw_curved['X'] = p_curved[:, 0]
    tw_curved['Y'] = p_curved[:, 1]
    tw_curved['Z'] = p_curved[:, 2]

    sv_curved_start = line.survey(element0='start',
                                    X0=sv_curved['X', 'start'],
                                    Y0=sv_curved['Y', 'start'],
                                    Z0=sv_curved['Z', 'start'],
                                    theta0=sv_curved['theta', 'start'],
                                    phi0=sv_curved['phi', 'start'],
                                    psi0=sv_curved['psi', 'start'])
    sv_curved_end = line.survey(element0='end',
                                    X0=sv_curved['X', 'end'],
                                    Y0=sv_curved['Y', 'end'],
                                    Z0=sv_curved['Z', 'end'],
                                    theta0=sv_curved['theta', 'end'],
                                    phi0=sv_curved['phi', 'end'],
                                    psi0=sv_curved['psi', 'end'])
    sv_no_slice_curved_start = line_no_slice.survey(element0='start',
                                    X0=sv_curved['X', 'start'],
                                    Y0=sv_curved['Y', 'start'],
                                    Z0=sv_curved['Z', 'start'],
                                    theta0=sv_curved['theta', 'start'],
                                    phi0=sv_curved['phi', 'start'],
                                    psi0=sv_curved['psi', 'start'])
    sv_no_slice_curved_end = line_no_slice.survey(element0='end',
                                    X0=sv_curved['X', 'end'],
                                    Y0=sv_curved['Y', 'end'],
                                    Z0=sv_curved['Z', 'end'],
                                    theta0=sv_curved['theta', 'end'],
                                    phi0=sv_curved['phi', 'end'],
                                    psi0=sv_curved['psi', 'end'])
    tw_no_slice_curved = line_no_slice.twiss(betx=1, bety=1)


    sv_straight.cols['s element_type angle']
    # is:
    # Table: 20 rows, 4 cols
    # name                      s element_type            angle
    # start                     0 Marker                          0
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
    # end                       5 Marker                          0
    # _end_point                5                                 0

    assert np.all(sv_straight['name'] ==[
        'start', '||drift_3::0', '||drift_4', 'mb_entry', 'mb..entry_map',
       'mb..0', 'mb..1', 'mb..2', 'mb..3', 'mid', 'mb..4', 'mb..5',
       'mb..6', 'mb..7', 'mb..exit_map', 'mb_exit', '||drift_5',
       '||drift_3::1', 'end', '_end_point'])

    # Assert entire columns using np.all
    assert np.all(sv_straight['element_type'] == [
        'Marker', 'Drift', 'Drift', 'Marker',
        'ThinSliceRBendEntry', 'ThickSliceRBend', 'ThickSliceRBend',
        'ThickSliceRBend', 'ThickSliceRBend', 'Marker', 'ThickSliceRBend',
        'ThickSliceRBend', 'ThickSliceRBend', 'ThickSliceRBend',
        'ThinSliceRBendExit', 'Marker', 'Drift', 'Drift',
        'Marker', ''])

    xo.assert_allclose(
        sv_straight['angle'],
        np.array([
            0.  , 0.  , 0.  , 0.  , 0.15, 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
            0.  , 0.  , 0.  , 0.15, 0.  , 0.  , 0.  , 0.  , 0.  ]),
        atol=1e-12
    )

    xo.assert_allclose(sv_straight['s'], np.array([
        0.       , 0.       , 0.5      , 0.9943602, 0.9943602, 0.9943602,
        1.       , 1.5      , 2.       , 2.5      , 2.5      , 3.       ,
        3.5      , 4.       , 4.0056398, 4.0056398, 4.0056398, 4.5      ,
        5.       , 5.       ]
    ), atol=1e-5)

    xo.assert_allclose(
        sv_straight['rot_s_rad'], 0, atol=5e-14)

    sv_straight.cols['X Y Z']
    # is:
    # SurveyTable: 20 rows, 4 cols
    # name                      X             Y             Z
    # start          -1.25496e-17     -0.261307      -2.48319
    # drift_1..0     -1.25496e-17     -0.261307      -2.48319
    # drift_1..1     -7.97441e-18     -0.186588      -1.98881
    # mb_entry       -3.45079e-18     -0.112711          -1.5
    # mb..entry_map  -3.45079e-18     -0.112711          -1.5
    # mb..0                     0    -0.0563557          -1.5
    # mb..1                     0    -0.0563557      -1.49438
    # mb..2                     0    -0.0563557     -0.996254
    # mb..3                     0    -0.0563557     -0.498127
    # mid                       0    -0.0563557             0
    # mb..4                     0    -0.0563557             0
    # mb..5                     0    -0.0563557      0.498127
    # mb..6                     0    -0.0563557      0.996254
    # mb..7                     0    -0.0563557       1.49438
    # mb..exit_map              0    -0.0563557           1.5
    # mb_exit        -3.45079e-18     -0.112711           1.5
    # drift_2..0     -3.45079e-18     -0.112711           1.5
    # drift_2..1     -7.97441e-18     -0.186588       1.98881
    # end            -1.25496e-17     -0.261307       2.48319

    xo.assert_allclose(sv_straight['Y'], 0, atol=5e-14)
    xo.assert_allclose(sv_straight['Z'], np.array([
        -2.48319461, -2.48319461, -1.98880907, -1.5       , -1.5       ,
        -1.5       , -1.49438132, -0.99625422, -0.49812711,  0.        ,
            0.        ,  0.49812711,  0.99625422,  1.49438132,  1.5       ,
            1.5       ,  1.5       ,  1.98880907,  2.48319461,  2.48319461]),
            atol=1e-8)
    xo.assert_allclose(sv_straight['X'], np.array([
        -0.26130674, -0.26130674, -0.18658768, -0.11271141, -0.11271141,
        -0.05635571, -0.05635571, -0.05635571, -0.05635571, -0.05635571,
        -0.05635571, -0.05635571, -0.05635571, -0.05635571, -0.05635571,
        -0.11271141, -0.11271141, -0.18658768, -0.26130674, -0.26130674]),
        atol=1e-8)


    sv_straight.cols['theta phi psi']
    # is:
    # SurveyTable: 20 rows, 4 cols
    # name                  theta           phi           psi
    # start                  0.15             0             0
    # drift_1..0             0.15             0             0
    # drift_1..1             0.15             0             0
    # mb_entry               0.15             0             0
    # mb..entry_map          0.15             0             0
    # mb..0                     0             0             0
    # mb..1                     0             0             0
    # mb..2                     0             0             0
    # mb..3                     0             0             0
    # mid                       0             0             0
    # mb..4                     0             0             0
    # mb..5                     0             0             0
    # mb..6                     0             0             0
    # mb..7                     0             0             0
    # mb..exit_map              0             0             0
    # mb_exit               -0.15             0             0
    # drift_2..0            -0.15             0             0
    # drift_2..1            -0.15             0             0
    # end                   -0.15             0             0
    # _end_point            -0.15             0             0

    xo.assert_allclose(sv_straight['phi'], 0, atol=5e-14)
    xo.assert_allclose(sv_straight['psi'], 0, atol=5e-14)
    xo.assert_allclose(sv_straight['theta'], np.array([
            0.15,  0.15,  0.15,  0.15,  0.15,  0.  ,  0.  ,  0.  ,  0.  ,
            0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  , -0.15, -0.15, -0.15,
        -0.15, -0.15]))


    sv_curved.cols['s element_type angle']
    # is:
    # Table: 20 rows, 4 cols
    # name                      s element_type            angle
    # start                     0 Marker                          0
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
    # end                       5 Marker                          0
    # _end_point                5                                 0

    assert np.all(sv_curved['name'] == [
        'start', '||drift_3::0', '||drift_4', 'mb_entry', 'mb..entry_map',
       'mb..0', 'mb..1', 'mb..2', 'mb..3', 'mid', 'mb..4', 'mb..5',
       'mb..6', 'mb..7', 'mb..exit_map', 'mb_exit', '||drift_5',
       '||drift_3::1', 'end', '_end_point'])

    assert np.all(sv_curved['element_type'] == [
        'Marker', 'Drift', 'Drift', 'Marker',
        'ThinSliceRBendEntry', 'ThickSliceRBend', 'ThickSliceRBend',
        'ThickSliceRBend', 'ThickSliceRBend', 'Marker', 'ThickSliceRBend',
        'ThickSliceRBend', 'ThickSliceRBend', 'ThickSliceRBend',
        'ThinSliceRBendExit', 'Marker', 'Drift', 'Drift',
        'Marker', ''])

    xo.assert_allclose(
        sv_curved['angle'],
        np.array([
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.00056187, 0.04981271, 0.04981271, 0.04981271, 0.        ,
        0.04981271, 0.04981271, 0.04981271, 0.00056187, 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ]),
        atol=1e-8
    )

    xo.assert_allclose(sv_curved['s'], np.array([
        0.       , 0.       , 0.5      , 0.9943602, 0.9943602, 0.9943602,
        1.       , 1.5      , 2.       , 2.5      , 2.5      , 3.       ,
        3.5      , 4.       , 4.0056398, 4.0056398, 4.0056398, 4.5      ,
        5.       , 5.       ]
    ), atol=1e-5)

    xo.assert_allclose(
        sv_curved['rot_s_rad'], 0, atol=5e-14)

    sv_curved.cols['X Y Z']
    # is:
    # SurveyTable: 20 rows, 4 cols
    # name                      X             Y             Z
    # start             -0.261307             0      -2.48319
    # drift_1..0        -0.261307             0      -2.48319
    # drift_1..1        -0.186588             0      -1.98881
    # mb_entry          -0.112711             0          -1.5
    # mb..entry_map     -0.112711             0          -1.5
    # mb..0             -0.112711             0          -1.5
    # mb..1              -0.11187             0      -1.49442
    # mb..2            -0.0497715             0     -0.998347
    # mb..3            -0.0124506             0     -0.499793
    # mid                       0             0             0
    # mb..4                     0             0             0
    # mb..5            -0.0124506             0      0.499793
    # mb..6            -0.0497715             0      0.998347
    # mb..7              -0.11187             0       1.49442
    # mb..exit_map      -0.112711             0           1.5
    # mb_exit           -0.112711             0           1.5
    # drift_2..0        -0.112711             0           1.5
    # drift_2..1        -0.186588             0       1.98881
    # end               -0.261307             0       2.48319
    # _end_point        -0.261307             0       2.48319

    xo.assert_allclose(sv_curved['Y'], 0, atol=5e-14)
    xo.assert_allclose(sv_curved['Z'], np.array([
        -2.48319461, -2.48319461, -1.98880907, -1.5       , -1.5       ,
        -1.5       , -1.49442329, -0.99834662, -0.49979325,  0.        ,
        0.        ,  0.49979325,  0.99834662,  1.49442329,  1.5       ,
        1.5       ,  1.5       ,  1.98880907,  2.48319461,  2.48319461
    ]), atol=1e-8)
    xo.assert_allclose(sv_curved['X'], np.array([
        -0.26130674, -0.26130674, -0.18658768, -0.11271141, -0.11271141,
        -0.11271141, -0.11187018, -0.04977152, -0.0124506 ,  0.        ,
        0.        , -0.0124506 , -0.04977152, -0.11187018, -0.11271141,
        -0.11271141, -0.11271141, -0.18658768, -0.26130674, -0.26130674
    ]), atol=1e-8)

    sv_curved.cols['theta phi psi']
    # is:
    # SurveyTable: 20 rows, 4 cols
    # name                  theta           phi           psi
    # start                  0.15             0             0
    # drift_1..0             0.15             0             0
    # drift_1..1             0.15             0             0
    # mb_entry               0.15             0             0
    # mb..entry_map          0.15             0             0
    # mb..0                  0.15             0             0
    # mb..1              0.149438             0             0
    # mb..2             0.0996254             0             0
    # mb..3             0.0498127             0             0
    # mid                       0             0             0
    # mb..4                     0             0             0
    # mb..5            -0.0498127             0             0
    # mb..6            -0.0996254             0             0
    # mb..7             -0.149438             0             0
    # mb..exit_map          -0.15             0             0
    # mb_exit               -0.15             0             0
    # drift_2..0            -0.15             0             0
    # drift_2..1            -0.15             0             0
    # end                   -0.15             0             0
    # _end_point            -0.15             0             0

    xo.assert_allclose(sv_curved['phi'], 0, atol=5e-14)
    xo.assert_allclose(sv_curved['psi'], 0, atol=5e-14)
    xo.assert_allclose(sv_curved['theta'], np.array([
            0.15      ,  0.15      ,  0.15      ,  0.15      ,  0.15      ,
            0.15      ,  0.14943813,  0.09962542,  0.04981271,  0.        ,
            0.        , -0.04981271, -0.09962542, -0.14943813, -0.15      ,
        -0.15      , -0.15      , -0.15      , -0.15      , -0.15      ],
        ), atol=1e-8)

    for nn in ['start', 'end']:
        xo.assert_allclose(sv_straight['X', nn], sv_curved['X', nn], atol=5e-14)
        xo.assert_allclose(sv_straight['Y', nn], sv_curved['Y', nn], atol=5e-14)
        xo.assert_allclose(sv_straight['Z', nn], sv_curved['Z', nn], atol=5e-14)
        xo.assert_allclose(sv_straight['theta', nn], sv_curved['theta', nn], atol=5e-14)
        xo.assert_allclose(sv_straight['phi', nn], sv_curved['phi', nn], atol=5e-14)
        xo.assert_allclose(sv_straight['psi', nn], sv_curved['psi', nn], atol=5e-14)

    xo.assert_allclose(sv_straight_start['X'], sv_straight['X'], atol=5e-14)
    xo.assert_allclose(sv_straight_start['Y'], sv_straight['Y'], atol=5e-14)
    xo.assert_allclose(sv_straight_start['Z'], sv_straight['Z'], atol=5e-14)
    xo.assert_allclose(sv_straight_start['theta'], sv_straight['theta'], atol=5e-14)
    xo.assert_allclose(sv_straight_start['phi'], sv_straight['phi'], atol=5e-14)
    xo.assert_allclose(sv_straight_start['psi'], sv_straight['psi'], atol=5e-14)

    xo.assert_allclose(sv_straight_end['X'], sv_straight['X'], atol=5e-14)
    xo.assert_allclose(sv_straight_end['Y'], sv_straight['Y'], atol=5e-14)
    xo.assert_allclose(sv_straight_end['Z'], sv_straight['Z'], atol=5e-14)
    xo.assert_allclose(sv_straight_end['theta'], sv_straight['theta'], atol=5e-14)
    xo.assert_allclose(sv_straight_end['phi'], sv_straight['phi'], atol=5e-14)
    xo.assert_allclose(sv_straight_end['psi'], sv_straight['psi'], atol=5e-14)

    xo.assert_allclose(sv_curved_start['X'], sv_curved['X'], atol=5e-14)
    xo.assert_allclose(sv_curved_start['Y'], sv_curved['Y'], atol=5e-14)
    xo.assert_allclose(sv_curved_start['Z'], sv_curved['Z'], atol=5e-14)
    xo.assert_allclose(sv_curved_start['theta'], sv_curved['theta'], atol=5e-14)
    xo.assert_allclose(sv_curved_start['phi'], sv_curved['phi'], atol=5e-14)
    xo.assert_allclose(sv_curved_start['psi'], sv_curved['psi'], atol=5e-14)

    xo.assert_allclose(sv_curved_end['X'], sv_curved['X'], atol=5e-14)
    xo.assert_allclose(sv_curved_end['Y'], sv_curved['Y'], atol=5e-14)
    xo.assert_allclose(sv_curved_end['Z'], sv_curved['Z'], atol=5e-14)
    xo.assert_allclose(sv_curved_end['theta'], sv_curved['theta'], atol=5e-14)
    xo.assert_allclose(sv_curved_end['phi'], sv_curved['phi'], atol=5e-14)
    xo.assert_allclose(sv_curved_end['psi'], sv_curved['psi'], atol=5e-14)
    xo.assert_allclose(tw_straight['X', 'mid'], 0, atol=5e-14)
    xo.assert_allclose(tw_straight['Y', 'mid'], 0, atol=5e-14)
    xo.assert_allclose(tw_straight['Z', 'mid'], 0, atol=5e-14)
    xo.assert_allclose(tw_curved['X', 'mid'], 0, atol=5e-14)
    xo.assert_allclose(tw_curved['Y', 'mid'], 0, atol=5e-14)
    xo.assert_allclose(tw_curved['Z', 'mid'], 0, atol=5e-14)
    xo.assert_allclose(tw_straight['X', 'mb_entry'], tw_curved['X', 'mb_entry'], atol=5e-14)
    xo.assert_allclose(tw_straight['Y', 'mb_entry'], tw_curved['Y', 'mb_entry'], atol=5e-14)
    xo.assert_allclose(tw_straight['Z', 'mb_entry'], tw_curved['Z', 'mb_entry'], atol=5e-14)
    xo.assert_allclose(tw_straight['X', 'mb_exit'], tw_curved['X', 'mb_exit'], atol=5e-14)
    xo.assert_allclose(tw_straight['Y', 'mb_exit'], tw_curved['Y', 'mb_exit'], atol=5e-14)
    xo.assert_allclose(tw_straight['Z', 'mb_exit'], tw_curved['Z', 'mb_exit'], atol=5e-14)

    xo.assert_allclose(tw_straight['x', 'mb_entry'], 0, atol=5e-14)
    xo.assert_allclose(tw_straight['y', 'mb_entry'], 0, atol=5e-14)
    xo.assert_allclose(tw_straight['x', 'mb_exit'], 0, atol=5e-14)
    xo.assert_allclose(tw_straight['y', 'mb_exit'], 0, atol=5e-14)
    xo.assert_allclose(tw_curved['x', 'mb_entry'], 0, atol=5e-14)
    xo.assert_allclose(tw_curved['y', 'mb_entry'], 0, atol=5e-14)
    xo.assert_allclose(tw_curved['x', 'mb_exit'], 0, atol=5e-14)
    xo.assert_allclose(tw_curved['y', 'mb_exit'], 0, atol=5e-14)

    xo.assert_allclose(tw_no_slice_curved['x', 'start'], tw_curved['x', 'start'],
                    atol=5e-14)
    xo.assert_allclose(tw_no_slice_curved['y', 'start'], tw_curved['y', 'start'],
                    atol=5e-14)
    xo.assert_allclose(tw_no_slice_curved['x', 'end'], tw_curved['x', 'end'],
                    atol=5e-14)
    xo.assert_allclose(tw_no_slice_curved['y', 'end'], tw_curved['y', 'end'],
                    atol=5e-14)
    xo.assert_allclose(tw_no_slice_straight['x', 'start'], tw_curved['x', 'start'],
                        atol=5e-14)
    xo.assert_allclose(tw_no_slice_straight['y', 'start'], tw_curved['y', 'start'],
                        atol=5e-14)
    xo.assert_allclose(tw_no_slice_straight['x', 'end'], tw_curved['x', 'end'],
                        atol=5e-14)
    xo.assert_allclose(tw_no_slice_straight['y', 'end'], tw_curved['y', 'end'],
                        atol=5e-14)

    for nn in ['start', 'end']:
        # Compare no_slice survey vs curved survey
        xo.assert_allclose(sv_no_slice_curved_start['X', nn], sv_curved['X', nn], atol=5e-14)
        xo.assert_allclose(sv_no_slice_curved_start['Y', nn], sv_curved['Y', nn], atol=5e-14)
        xo.assert_allclose(sv_no_slice_curved_start['Z', nn], sv_curved['Z', nn], atol=5e-14)
        xo.assert_allclose(sv_no_slice_curved_start['theta', nn], sv_curved['theta', nn], atol=5e-14)
        xo.assert_allclose(sv_no_slice_curved_start['phi', nn], sv_curved['phi', nn], atol=5e-14)
        xo.assert_allclose(sv_no_slice_curved_start['psi', nn], sv_curved['psi', nn], atol=5e-14)
        xo.assert_allclose(sv_no_slice_curved_start['s', nn], sv_curved['s', nn], atol=5e-14)

        xo.assert_allclose(sv_no_slice_curved_end['X', nn], sv_curved['X', nn], atol=5e-14)
        xo.assert_allclose(sv_no_slice_curved_end['Y', nn], sv_curved['Y', nn], atol=5e-14)
        xo.assert_allclose(sv_no_slice_curved_end['Z', nn], sv_curved['Z', nn], atol=5e-14)
        xo.assert_allclose(sv_no_slice_curved_end['theta', nn], sv_curved['theta', nn], atol=5e-14)
        xo.assert_allclose(sv_no_slice_curved_end['phi', nn], sv_curved['phi', nn], atol=5e-14)
        xo.assert_allclose(sv_no_slice_curved_end['psi', nn], sv_curved['psi', nn], atol=5e-14)
        xo.assert_allclose(sv_no_slice_curved_end['s', nn], sv_curved['s', nn], atol=5e-14)

    for nn in ['start', 'end']:
        # Compare no_slice survey vs straight survey
        xo.assert_allclose(sv_no_slice_straight_start['X', nn], sv_straight['X', nn], atol=5e-14)
        xo.assert_allclose(sv_no_slice_straight_start['Y', nn], sv_straight['Y', nn], atol=5e-14)
        xo.assert_allclose(sv_no_slice_straight_start['Z', nn], sv_straight['Z', nn], atol=5e-14)
        xo.assert_allclose(sv_no_slice_straight_start['theta', nn], sv_straight['theta', nn], atol=5e-14)
        xo.assert_allclose(sv_no_slice_straight_start['phi', nn], sv_straight['phi', nn], atol=5e-14)
        xo.assert_allclose(sv_no_slice_straight_start['psi', nn], sv_straight['psi', nn], atol=5e-14)
        xo.assert_allclose(sv_no_slice_straight_start['s', nn], sv_straight['s', nn], atol=5e-14)

        xo.assert_allclose(sv_no_slice_straight_end['X', nn], sv_straight['X', nn], atol=5e-14)
        xo.assert_allclose(sv_no_slice_straight_end['Y', nn], sv_straight['Y', nn], atol=5e-14)
        xo.assert_allclose(sv_no_slice_straight_end['Z', nn], sv_straight['Z', nn], atol=5e-14)
        xo.assert_allclose(sv_no_slice_straight_end['theta', nn], sv_straight['theta', nn], atol=5e-14)
        xo.assert_allclose(sv_no_slice_straight_end['phi', nn], sv_straight['phi', nn], atol=5e-14)
        xo.assert_allclose(sv_no_slice_straight_end['psi', nn], sv_straight['psi', nn], atol=5e-14)
        xo.assert_allclose(sv_no_slice_straight_end['s', nn], sv_straight['s', nn], atol=5e-14)

def test_rbend_straight_survey_h_angle_diff():

    env = xt.Environment(particle_ref=xt.Particles(p0c=10e9))

    edge_model = 'full'

    line = env.new_line(length=5, components=[
        env.new('mb', 'RBend', angle=0.3, k0_from_h=True, length_straight=3,
                rbend_angle_diff=0.3,
                model='bend-kick-bend',
                rbend_model='straight-body', edge_entry_model=edge_model, edge_exit_model=edge_model,
                anchor='start', at=1.)])
    line['mb'].rbend_compensate_sagitta = False
    line['mb'].rbend_shift = line['mb']._x0_in
    line.insert('start', xt.Marker(), at=0)
    line.append('end', xt.Marker())

    line_no_slice = line.copy(shallow=True)

    line.slice_thick_elements(
        slicing_strategies=[
                xt.Strategy(slicing=xt.Uniform(6, mode='thick'))])

    line['mb'].rbend_model = 'straight-body'
    sv_straight = line.survey()
    tt_straight = line.get_table(attr=True)
    tw_straight = line.twiss(betx=1, bety=1)
    p_straight = (sv_straight.p0 + tw_straight.x[:, None] * sv_straight['ex']
                                + tw_straight.y[:, None] * sv_straight['ey'])
    tw_straight['X'] = p_straight[:, 0]
    tw_straight['Y'] = p_straight[:, 1]
    tw_straight['Z'] = p_straight[:, 2]

    sv_straight_start = line.survey(element0='start',
                                    X0=sv_straight['X', 'start'],
                                    Y0=sv_straight['Y', 'start'],
                                    Z0=sv_straight['Z', 'start'],
                                    theta0=sv_straight['theta', 'start'],
                                    phi0=sv_straight['phi', 'start'],
                                    psi0=sv_straight['psi', 'start'])
    sv_straight_end = line.survey(element0='end',
                                    X0=sv_straight['X', 'end'],
                                    Y0=sv_straight['Y', 'end'],
                                    Z0=sv_straight['Z', 'end'],
                                    theta0=sv_straight['theta', 'end'],
                                    phi0=sv_straight['phi', 'end'],
                                    psi0=sv_straight['psi', 'end'])

    sv_no_slice_start = line_no_slice.survey(element0='start',
                                    X0=sv_straight['X', 'start'],
                                    Y0=sv_straight['Y', 'start'],
                                    Z0=sv_straight['Z', 'start'],
                                    theta0=sv_straight['theta', 'start'],
                                    phi0=sv_straight['phi', 'start'],
                                    psi0=sv_straight['psi', 'start'])
    sv_no_slice_end = line_no_slice.survey(element0='end',
                                    X0=sv_straight['X', 'end'],
                                    Y0=sv_straight['Y', 'end'],
                                    Z0=sv_straight['Z', 'end'],
                                    theta0=sv_straight['theta', 'end'],
                                    phi0=sv_straight['phi', 'end'],
                                    psi0=sv_straight['psi', 'end'])
    tw_no_slice_straight = line_no_slice.twiss(betx=1, bety=1)

    line['mb'].rbend_model = 'curved-body'
    sv_curved = line.survey()
    tt_curved = line.get_table(attr=True)
    tw_curved = line.twiss(betx=1, bety=1)
    p_curved = (sv_curved.p0 + tw_curved.x[:, None] * sv_curved['ex']
                            + tw_curved.y[:, None] * sv_curved['ey'])
    tw_curved['X'] = p_curved[:, 0]
    tw_curved['Y'] = p_curved[:, 1]
    tw_curved['Z'] = p_curved[:, 2]

    sv_curved_start = line.survey(element0='start',
                                    X0=sv_curved['X', 'start'],
                                    Y0=sv_curved['Y', 'start'],
                                    Z0=sv_curved['Z', 'start'],
                                    theta0=sv_curved['theta', 'start'],
                                    phi0=sv_curved['phi', 'start'],
                                    psi0=sv_curved['psi', 'start'])
    sv_curved_end = line.survey(element0='end',
                                    X0=sv_curved['X', 'end'],
                                    Y0=sv_curved['Y', 'end'],
                                    Z0=sv_curved['Z', 'end'],
                                    theta0=sv_curved['theta', 'end'],
                                    phi0=sv_curved['phi', 'end'],
                                    psi0=sv_curved['psi', 'end'])
    sv_no_slice_curved_start = line_no_slice.survey(element0='start',
                                    X0=sv_curved['X', 'start'],
                                    Y0=sv_curved['Y', 'start'],
                                    Z0=sv_curved['Z', 'start'],
                                    theta0=sv_curved['theta', 'start'],
                                    phi0=sv_curved['phi', 'start'],
                                    psi0=sv_curved['psi', 'start'])
    sv_no_slice_curved_end = line_no_slice.survey(element0='end',
                                    X0=sv_curved['X', 'end'],
                                    Y0=sv_curved['Y', 'end'],
                                    Z0=sv_curved['Z', 'end'],
                                    theta0=sv_curved['theta', 'end'],
                                    phi0=sv_curved['phi', 'end'],
                                    psi0=sv_curved['psi', 'end'])
    tw_no_slice_curved = line_no_slice.twiss(betx=1, bety=1)


    assert np.all(sv_straight['name'] == [
        'start', '||drift_1', 'mb_entry', 'mb..entry_map', 'mb..0',
        'mb..1', 'mb..2', 'mb..3', 'mb..4', 'mb..5', 'mb..exit_map',
        'mb_exit', '||drift_2', 'end', '_end_point'])

    # Assert entire columns using np.all
    assert np.all(sv_straight['element_type'] == [
        'Marker', 'Drift', 'Marker', 'ThinSliceRBendEntry',
        'ThickSliceRBend', 'ThickSliceRBend', 'ThickSliceRBend',
        'ThickSliceRBend', 'ThickSliceRBend', 'ThickSliceRBend',
        'ThinSliceRBendExit', 'Marker', 'Drift', 'Marker', ''])

    xo.assert_allclose(
        sv_straight['angle'],
        np.array([
        0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.3, 0. , 0. ,
        0. , 0. ]),
        atol=1e-12
    )

    xo.assert_allclose(sv_straight['s'], np.array([
        0.        , 0.        , 1.        , 1.        , 1.        ,
        1.5075795 , 2.01515901, 2.52273851, 3.03031802, 3.53789752,
        4.04547703, 4.04547703, 4.04547703, 5.        , 5.        ]
    ), atol=1e-5)

    xo.assert_allclose(
        sv_straight['rot_s_rad'], 0, atol=1e-14)

    sv_straight.cols['X Y Z']
    xo.assert_allclose(sv_straight['Y'], 0, atol=1e-14)
    xo.assert_allclose(sv_straight['Z'], np.array(
        [0.        , 0.        , 1.        , 1.        , 1.        ,
        1.5       , 2.        , 2.5       , 3.        , 3.5       ,
        4.        , 4.        , 4.        , 4.91189063, 4.91189063]),
        atol=1e-8)
    xo.assert_allclose(sv_straight['X'], np.array([
            0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        -1.38777878e-17, -1.38777878e-17, -1.38777878e-17, -1.38777878e-17,
        -1.38777878e-17, -1.38777878e-17, -1.38777878e-17, -4.53405654e-01,
        -4.53405654e-01, -7.35486481e-01, -7.35486481e-01]),
        atol=1e-8)


    sv_straight.cols['theta phi psi']
    xo.assert_allclose(sv_straight['phi'], 0, atol=1e-14)
    xo.assert_allclose(sv_straight['psi'], 0, atol=1e-14)
    xo.assert_allclose(sv_straight['theta'], np.array([
            0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
        -0.3, -0.3, -0.3, -0.3]))


    sv_curved.cols['s element_type angle']
    assert np.all(sv_curved['name'] == [
        'start', '||drift_1', 'mb_entry', 'mb..entry_map', 'mb..0',
        'mb..1', 'mb..2', 'mb..3', 'mb..4', 'mb..5', 'mb..exit_map',
        'mb_exit', '||drift_2', 'end', '_end_point'])

    assert np.all(sv_curved['element_type'] == [
        'Marker', 'Drift', 'Marker', 'ThinSliceRBendEntry',
        'ThickSliceRBend', 'ThickSliceRBend', 'ThickSliceRBend',
        'ThickSliceRBend', 'ThickSliceRBend', 'ThickSliceRBend',
        'ThinSliceRBendExit', 'Marker', 'Drift', 'Marker', ''])

    xo.assert_allclose(
        sv_curved['angle'],
        np.array([
        0.  , 0.  , 0.  , 0.  , 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.  ,
        0.  , 0.  , 0.  , 0.  ]),
        atol=1e-8
    )

    xo.assert_allclose(sv_curved['s'], np.array([
        0.        , 0.        , 1.        , 1.        , 1.        ,
        1.5075795 , 2.01515901, 2.52273851, 3.03031802, 3.53789752,
        4.04547703, 4.04547703, 4.04547703, 5.        , 5.        ]
    ), atol=1e-5)

    xo.assert_allclose(
        sv_curved['rot_s_rad'], 0, atol=1e-14)

    sv_curved.cols['X Y Z']
    xo.assert_allclose(sv_curved['Y'], 0, atol=1e-14)
    xo.assert_allclose(sv_curved['Z'], np.array([
        0.      , 0.      , 1.      , 1.      , 1.      , 1.507368,
        2.013468, 2.517035, 3.01681 , 3.511544, 4.      , 4.      ,
        4.      , 4.911891, 4.911891]), atol=1e-5)
    xo.assert_allclose(sv_curved['X'], np.array(
        [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        -0.01268684, -0.05071567, -0.11399141, -0.20235593, -0.31558835,
        -0.45340565, -0.45340565, -0.45340565, -0.73548648, -0.73548648]),
        atol=1e-8)

    sv_curved.cols['theta phi psi']
    xo.assert_allclose(sv_curved['phi'], 0, atol=1e-14)
    xo.assert_allclose(sv_curved['psi'], 0, atol=1e-14)
    xo.assert_allclose(sv_curved['theta'], np.array([
            0.  ,  0.  ,  0.  ,  0.  ,  0.  , -0.05, -0.1 , -0.15, -0.2 ,
        -0.25, -0.3 , -0.3 , -0.3 , -0.3 , -0.3 ],
        ), atol=1e-8)

    for nn in ['start', 'end']:
        xo.assert_allclose(sv_straight['X', nn], sv_curved['X', nn], atol=1e-14)
        xo.assert_allclose(sv_straight['Y', nn], sv_curved['Y', nn], atol=1e-14)
        xo.assert_allclose(sv_straight['Z', nn], sv_curved['Z', nn], atol=1e-14)
        xo.assert_allclose(sv_straight['theta', nn], sv_curved['theta', nn], atol=1e-14)
        xo.assert_allclose(sv_straight['phi', nn], sv_curved['phi', nn], atol=1e-14)
        xo.assert_allclose(sv_straight['psi', nn], sv_curved['psi', nn], atol=1e-14)

    xo.assert_allclose(sv_straight_start['X'], sv_straight['X'], atol=1e-14)
    xo.assert_allclose(sv_straight_start['Y'], sv_straight['Y'], atol=1e-14)
    xo.assert_allclose(sv_straight_start['Z'], sv_straight['Z'], atol=1e-14)
    xo.assert_allclose(sv_straight_start['theta'], sv_straight['theta'], atol=1e-14)
    xo.assert_allclose(sv_straight_start['phi'], sv_straight['phi'], atol=1e-14)
    xo.assert_allclose(sv_straight_start['psi'], sv_straight['psi'], atol=1e-14)

    xo.assert_allclose(sv_straight_end['X'], sv_straight['X'], atol=1e-14)
    xo.assert_allclose(sv_straight_end['Y'], sv_straight['Y'], atol=1e-14)
    xo.assert_allclose(sv_straight_end['Z'], sv_straight['Z'], atol=1e-14)
    xo.assert_allclose(sv_straight_end['theta'], sv_straight['theta'], atol=1e-14)
    xo.assert_allclose(sv_straight_end['phi'], sv_straight['phi'], atol=1e-14)
    xo.assert_allclose(sv_straight_end['psi'], sv_straight['psi'], atol=1e-14)

    xo.assert_allclose(sv_curved_start['X'], sv_curved['X'], atol=1e-14)
    xo.assert_allclose(sv_curved_start['Y'], sv_curved['Y'], atol=1e-14)
    xo.assert_allclose(sv_curved_start['Z'], sv_curved['Z'], atol=1e-14)
    xo.assert_allclose(sv_curved_start['theta'], sv_curved['theta'], atol=1e-14)
    xo.assert_allclose(sv_curved_start['phi'], sv_curved['phi'], atol=1e-14)
    xo.assert_allclose(sv_curved_start['psi'], sv_curved['psi'], atol=1e-14)

    xo.assert_allclose(sv_curved_end['X'], sv_curved['X'], atol=1e-14)
    xo.assert_allclose(sv_curved_end['Y'], sv_curved['Y'], atol=1e-14)
    xo.assert_allclose(sv_curved_end['Z'], sv_curved['Z'], atol=1e-14)
    xo.assert_allclose(sv_curved_end['theta'], sv_curved['theta'], atol=1e-14)
    xo.assert_allclose(sv_curved_end['phi'], sv_curved['phi'], atol=1e-14)
    xo.assert_allclose(sv_curved_end['psi'], sv_curved['psi'], atol=1e-14)
    xo.assert_allclose(tw_straight['X', 'start'], 0, atol=1e-14)
    xo.assert_allclose(tw_straight['Y', 'start'], 0, atol=1e-14)
    xo.assert_allclose(tw_straight['Z', 'start'], 0, atol=1e-14)
    xo.assert_allclose(tw_curved['X', 'start'], 0, atol=1e-14)
    xo.assert_allclose(tw_curved['Y', 'start'], 0, atol=1e-14)
    xo.assert_allclose(tw_curved['Z', 'start'], 0, atol=1e-14)
    xo.assert_allclose(tw_straight['X', 'mb_entry'], tw_curved['X', 'mb_entry'], atol=1e-14)
    xo.assert_allclose(tw_straight['Y', 'mb_entry'], tw_curved['Y', 'mb_entry'], atol=1e-14)
    xo.assert_allclose(tw_straight['Z', 'mb_entry'], tw_curved['Z', 'mb_entry'], atol=1e-14)
    xo.assert_allclose(tw_straight['X', 'mb_exit'], tw_curved['X', 'mb_exit'], atol=1e-14)
    xo.assert_allclose(tw_straight['Y', 'mb_exit'], tw_curved['Y', 'mb_exit'], atol=1e-14)
    xo.assert_allclose(tw_straight['Z', 'mb_exit'], tw_curved['Z', 'mb_exit'], atol=1e-14)

    xo.assert_allclose(tw_straight['x', 'mb_entry'], 0, atol=1e-14)
    xo.assert_allclose(tw_straight['y', 'mb_entry'], 0, atol=1e-14)
    xo.assert_allclose(tw_straight['x', 'mb_exit'], 0, atol=1e-14)
    xo.assert_allclose(tw_straight['y', 'mb_exit'], 0, atol=1e-14)
    xo.assert_allclose(tw_curved['x', 'mb_entry'], 0, atol=1e-14)
    xo.assert_allclose(tw_curved['y', 'mb_entry'], 0, atol=1e-14)
    xo.assert_allclose(tw_curved['x', 'mb_exit'], 0, atol=1e-14)
    xo.assert_allclose(tw_curved['y', 'mb_exit'], 0, atol=1e-14)

    xo.assert_allclose(tw_no_slice_curved['x', 'start'], tw_curved['x', 'start'],
                    atol=1e-14)
    xo.assert_allclose(tw_no_slice_curved['y', 'start'], tw_curved['y', 'start'],
                    atol=1e-14)
    xo.assert_allclose(tw_no_slice_curved['x', 'end'], tw_curved['x', 'end'],
                    atol=1e-14)
    xo.assert_allclose(tw_no_slice_curved['y', 'end'], tw_curved['y', 'end'],
                    atol=1e-14)
    xo.assert_allclose(tw_no_slice_straight['x', 'start'], tw_curved['x', 'start'],
                        atol=1e-14)
    xo.assert_allclose(tw_no_slice_straight['y', 'start'], tw_curved['y', 'start'],
                        atol=1e-14)
    xo.assert_allclose(tw_no_slice_straight['x', 'end'], tw_curved['x', 'end'],
                        atol=1e-14)
    xo.assert_allclose(tw_no_slice_straight['y', 'end'], tw_curved['y', 'end'],
                        atol=1e-14)

    for nn in ['start', 'end']:
        # Compare no_slice survey vs curved survey
        xo.assert_allclose(sv_no_slice_curved_start['X', nn], sv_curved['X', nn], atol=1e-14)
        xo.assert_allclose(sv_no_slice_curved_start['Y', nn], sv_curved['Y', nn], atol=1e-14)
        xo.assert_allclose(sv_no_slice_curved_start['Z', nn], sv_curved['Z', nn], atol=1e-14)
        xo.assert_allclose(sv_no_slice_curved_start['theta', nn], sv_curved['theta', nn], atol=1e-14)
        xo.assert_allclose(sv_no_slice_curved_start['phi', nn], sv_curved['phi', nn], atol=1e-14)
        xo.assert_allclose(sv_no_slice_curved_start['psi', nn], sv_curved['psi', nn], atol=1e-14)
        xo.assert_allclose(sv_no_slice_curved_start['s', nn], sv_curved['s', nn], atol=1e-14)

        xo.assert_allclose(sv_no_slice_curved_end['X', nn], sv_curved['X', nn], atol=1e-14)
        xo.assert_allclose(sv_no_slice_curved_end['Y', nn], sv_curved['Y', nn], atol=1e-14)
        xo.assert_allclose(sv_no_slice_curved_end['Z', nn], sv_curved['Z', nn], atol=1e-14)
        xo.assert_allclose(sv_no_slice_curved_end['theta', nn], sv_curved['theta', nn], atol=1e-14)
        xo.assert_allclose(sv_no_slice_curved_end['phi', nn], sv_curved['phi', nn], atol=1e-14)
        xo.assert_allclose(sv_no_slice_curved_end['psi', nn], sv_curved['psi', nn], atol=1e-14)
        xo.assert_allclose(sv_no_slice_curved_end['s', nn], sv_curved['s', nn], atol=1e-14)

def test_rbend_straight_body_survey_v():

    env = xt.Environment(particle_ref=xt.Particles(p0c=10e9))

    edge_model = 'full'

    line = env.new_line(length=5, components=[
        env.new('mb', 'RBend', angle=0.3, k0_from_h=True, length_straight=3,
                rot_s_rad=np.pi/2,
                model='bend-kick-bend',
                rbend_model='straight-body', edge_entry_model=edge_model, edge_exit_model=edge_model,
                at=2.5)])
    line.insert('start', xt.Marker(), at=0)
    line.append('end', xt.Marker())

    line_no_slice = line.copy(shallow=True)

    line.cut_at_s(np.linspace(0, line.get_length(), 11))
    line.insert('mid', xt.Marker(), at=2.5)

    line['mb'].rbend_model = 'straight-body'
    sv_straight = line.survey(element0='mid', Y0=-line['mb'].sagitta/2)
    tt_straight = line.get_table(attr=True)
    tw_straight = line.twiss(betx=1, bety=1)
    p_straight = (sv_straight.p0 + tw_straight.x[:, None] * sv_straight['ex']
                                + tw_straight.y[:, None] * sv_straight['ey'])
    tw_straight['X'] = p_straight[:, 0]
    tw_straight['Y'] = p_straight[:, 1]
    tw_straight['Z'] = p_straight[:, 2]

    sv_straight_start = line.survey(element0='start',
                                    X0=sv_straight['X', 'start'],
                                    Y0=sv_straight['Y', 'start'],
                                    Z0=sv_straight['Z', 'start'],
                                    theta0=sv_straight['theta', 'start'],
                                    phi0=sv_straight['phi', 'start'],
                                    psi0=sv_straight['psi', 'start'])
    sv_straight_end = line.survey(element0='end',
                                    X0=sv_straight['X', 'end'],
                                    Y0=sv_straight['Y', 'end'],
                                    Z0=sv_straight['Z', 'end'],
                                    theta0=sv_straight['theta', 'end'],
                                    phi0=sv_straight['phi', 'end'],
                                    psi0=sv_straight['psi', 'end'])

    sv_no_slice_straight_start = line_no_slice.survey(element0='start',
                                    X0=sv_straight['X', 'start'],
                                    Y0=sv_straight['Y', 'start'],
                                    Z0=sv_straight['Z', 'start'],
                                    theta0=sv_straight['theta', 'start'],
                                    phi0=sv_straight['phi', 'start'],
                                    psi0=sv_straight['psi', 'start'])
    sv_no_slice_straight_end = line_no_slice.survey(element0='end',
                                    X0=sv_straight['X', 'end'],
                                    Y0=sv_straight['Y', 'end'],
                                    Z0=sv_straight['Z', 'end'],
                                    theta0=sv_straight['theta', 'end'],
                                    phi0=sv_straight['phi', 'end'],
                                    psi0=sv_straight['psi', 'end'])
    tw_no_slice_straight = line_no_slice.twiss(betx=1, bety=1)

    line['mb'].rbend_model = 'curved-body'
    sv_curved = line.survey(element0='mid')
    tt_curved = line.get_table(attr=True)
    tw_curved = line.twiss(betx=1, bety=1)
    p_curved = (sv_curved.p0 + tw_curved.x[:, None] * sv_curved['ex']
                            + tw_curved.y[:, None] * sv_curved['ey'])
    tw_curved['X'] = p_curved[:, 0]
    tw_curved['Y'] = p_curved[:, 1]
    tw_curved['Z'] = p_curved[:, 2]

    sv_curved_start = line.survey(element0='start',
                                    X0=sv_curved['X', 'start'],
                                    Y0=sv_curved['Y', 'start'],
                                    Z0=sv_curved['Z', 'start'],
                                    theta0=sv_curved['theta', 'start'],
                                    phi0=sv_curved['phi', 'start'],
                                    psi0=sv_curved['psi', 'start'])
    sv_curved_end = line.survey(element0='end',
                                    X0=sv_curved['X', 'end'],
                                    Y0=sv_curved['Y', 'end'],
                                    Z0=sv_curved['Z', 'end'],
                                    theta0=sv_curved['theta', 'end'],
                                    phi0=sv_curved['phi', 'end'],
                                    psi0=sv_curved['psi', 'end'])
    sv_no_slice_curved_start = line_no_slice.survey(element0='start',
                                    X0=sv_curved['X', 'start'],
                                    Y0=sv_curved['Y', 'start'],
                                    Z0=sv_curved['Z', 'start'],
                                    theta0=sv_curved['theta', 'start'],
                                    phi0=sv_curved['phi', 'start'],
                                    psi0=sv_curved['psi', 'start'])
    sv_no_slice_curved_end = line_no_slice.survey(element0='end',
                                    X0=sv_curved['X', 'end'],
                                    Y0=sv_curved['Y', 'end'],
                                    Z0=sv_curved['Z', 'end'],
                                    theta0=sv_curved['theta', 'end'],
                                    phi0=sv_curved['phi', 'end'],
                                    psi0=sv_curved['psi', 'end'])
    tw_no_slice_curved = line_no_slice.twiss(betx=1, bety=1)


    sv_straight.cols['s element_type angle']
    # is:
    # Table: 20 rows, 4 cols
    # name                      s element_type            angle
    # start                     0 Marker                          0
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
    # end                       5 Marker                          0
    # _end_point                5                                 0

    assert np.all(sv_straight['name'] == [
        'start', '||drift_3::0', '||drift_4', 'mb_entry', 'mb..entry_map',
       'mb..0', 'mb..1', 'mb..2', 'mb..3', 'mid', 'mb..4', 'mb..5',
       'mb..6', 'mb..7', 'mb..exit_map', 'mb_exit', '||drift_5',
       '||drift_3::1', 'end', '_end_point'])

    # Assert entire columns using np.all
    assert np.all(sv_straight['element_type'] == [
        'Marker', 'Drift', 'Drift', 'Marker',
        'ThinSliceRBendEntry', 'ThickSliceRBend', 'ThickSliceRBend',
        'ThickSliceRBend', 'ThickSliceRBend', 'Marker', 'ThickSliceRBend',
        'ThickSliceRBend', 'ThickSliceRBend', 'ThickSliceRBend',
        'ThinSliceRBendExit', 'Marker', 'Drift', 'Drift',
        'Marker', ''])

    xo.assert_allclose(
        sv_straight['angle'],
        np.array([
            0.  , 0.  , 0.  , 0.  , 0.15, 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
            0.  , 0.  , 0.  , 0.15, 0.  , 0.  , 0.  , 0.  , 0.  ]),
        atol=1e-12
    )

    xo.assert_allclose(sv_straight['s'], np.array([
        0.       , 0.       , 0.5      , 0.9943602, 0.9943602, 0.9943602,
        1.       , 1.5      , 2.       , 2.5      , 2.5      , 3.       ,
        3.5      , 4.       , 4.0056398, 4.0056398, 4.0056398, 4.5      ,
        5.       , 5.       ]
    ), atol=1e-5)

    xo.assert_allclose(
        sv_straight['rot_s_rad'],
        np.array([
            0.        , 0.        , 0.        , 0.        , 1.57079633,
            1.57079633, 1.57079633, 1.57079633, 1.57079633, 0.        ,
            1.57079633, 1.57079633, 1.57079633, 1.57079633, 1.57079633,
            0.        , 0.        , 0.        , 0.        , 0.        ]),
        atol=1e-8
    )

    sv_straight.cols['X Y Z']
    # is:
    # SurveyTable: 20 rows, 4 cols
    # name                      X             Y             Z
    # start          -1.25496e-17     -0.261307      -2.48319
    # drift_1..0     -1.25496e-17     -0.261307      -2.48319
    # drift_1..1     -7.97441e-18     -0.186588      -1.98881
    # mb_entry       -3.45079e-18     -0.112711          -1.5
    # mb..entry_map  -3.45079e-18     -0.112711          -1.5
    # mb..0                     0    -0.0563557          -1.5
    # mb..1                     0    -0.0563557      -1.49438
    # mb..2                     0    -0.0563557     -0.996254
    # mb..3                     0    -0.0563557     -0.498127
    # mid                       0    -0.0563557             0
    # mb..4                     0    -0.0563557             0
    # mb..5                     0    -0.0563557      0.498127
    # mb..6                     0    -0.0563557      0.996254
    # mb..7                     0    -0.0563557       1.49438
    # mb..exit_map              0    -0.0563557           1.5
    # mb_exit        -3.45079e-18     -0.112711           1.5
    # drift_2..0     -3.45079e-18     -0.112711           1.5
    # drift_2..1     -7.97441e-18     -0.186588       1.98881
    # end            -1.25496e-17     -0.261307       2.48319

    xo.assert_allclose(sv_straight['X'], 0, atol=5e-14)
    xo.assert_allclose(sv_straight['Z'], np.array([
        -2.48319461, -2.48319461, -1.98880907, -1.5       , -1.5       ,
        -1.5       , -1.49438132, -0.99625422, -0.49812711,  0.        ,
            0.        ,  0.49812711,  0.99625422,  1.49438132,  1.5       ,
            1.5       ,  1.5       ,  1.98880907,  2.48319461,  2.48319461]),
            atol=1e-8)
    xo.assert_allclose(sv_straight['Y'], np.array([
        -0.26130674, -0.26130674, -0.18658768, -0.11271141, -0.11271141,
        -0.05635571, -0.05635571, -0.05635571, -0.05635571, -0.05635571,
        -0.05635571, -0.05635571, -0.05635571, -0.05635571, -0.05635571,
        -0.11271141, -0.11271141, -0.18658768, -0.26130674, -0.26130674]),
        atol=1e-8)


    sv_straight.cols['theta phi psi']
    # is:
    # SurveyTable: 20 rows, 4 cols
    # name                  theta           phi           psi
    # start           9.25436e-18          0.15  -6.95382e-19
    # drift_1..0      9.25436e-18          0.15  -6.95382e-19
    # drift_1..1      9.25436e-18          0.15  -6.95382e-19
    # mb_entry        9.25436e-18          0.15  -6.95382e-19
    # mb..entry_map   9.25436e-18          0.15  -6.95382e-19
    # mb..0                     0             0             0
    # mb..1                     0             0             0
    # mb..2                     0             0             0
    # mb..3                     0             0             0
    # mid                       0             0             0
    # mb..4                     0             0             0
    # mb..5                     0             0             0
    # mb..6                     0             0             0
    # mb..7                     0             0             0
    # mb..exit_map              0             0             0
    # mb_exit        -9.25436e-18         -0.15  -6.95382e-19
    # drift_2..0     -9.25436e-18         -0.15  -6.95382e-19
    # drift_2..1     -9.25436e-18         -0.15  -6.95382e-19
    # end            -9.25436e-18         -0.15  -6.95382e-19

    xo.assert_allclose(sv_straight['theta'], 0, atol=5e-14)
    xo.assert_allclose(sv_straight['psi'], 0, atol=5e-14)
    xo.assert_allclose(sv_straight['phi'], np.array([
            0.15,  0.15,  0.15,  0.15,  0.15,  0.  ,  0.  ,  0.  ,  0.  ,
            0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  , -0.15, -0.15, -0.15,
        -0.15, -0.15]))


    sv_curved.cols['s element_type angle']
    # is:
    # Table: 20 rows, 4 cols
    # name                      s element_type            angle
    # start                     0 Marker                          0
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
    # end                       5 Marker                          0
    # _end_point                5                                 0

    assert np.all(sv_curved['name'] == 
        ['start', '||drift_3::0', '||drift_4', 'mb_entry', 'mb..entry_map',
       'mb..0', 'mb..1', 'mb..2', 'mb..3', 'mid', 'mb..4', 'mb..5',
       'mb..6', 'mb..7', 'mb..exit_map', 'mb_exit', '||drift_5',
       '||drift_3::1', 'end', '_end_point'])

    assert np.all(sv_curved['element_type'] == [
        'Marker', 'Drift', 'Drift', 'Marker',
        'ThinSliceRBendEntry', 'ThickSliceRBend', 'ThickSliceRBend',
        'ThickSliceRBend', 'ThickSliceRBend', 'Marker', 'ThickSliceRBend',
        'ThickSliceRBend', 'ThickSliceRBend', 'ThickSliceRBend',
        'ThinSliceRBendExit', 'Marker', 'Drift', 'Drift',
        'Marker', ''])

    xo.assert_allclose(
        sv_curved['angle'],
        np.array([
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.00056187, 0.04981271, 0.04981271, 0.04981271, 0.        ,
        0.04981271, 0.04981271, 0.04981271, 0.00056187, 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ]),
        atol=1e-8
    )

    xo.assert_allclose(sv_curved['s'], np.array([
        0.       , 0.       , 0.5      , 0.9943602, 0.9943602, 0.9943602,
        1.       , 1.5      , 2.       , 2.5      , 2.5      , 3.       ,
        3.5      , 4.       , 4.0056398, 4.0056398, 4.0056398, 4.5      ,
        5.       , 5.       ]
    ), atol=1e-5)

    xo.assert_allclose(
        sv_curved['rot_s_rad'],
        np.array([
            0.        , 0.        , 0.        , 0.        , 1.57079633,
            1.57079633, 1.57079633, 1.57079633, 1.57079633, 0.        ,
            1.57079633, 1.57079633, 1.57079633, 1.57079633, 1.57079633,
            0.        , 0.        , 0.        , 0.        , 0.        ]),
        atol=1e-8
    )

    sv_curved.cols['X Y Z']
    # is:
    # SurveyTable: 20 rows, 4 cols
    # name                      X             Y             Z
    # start          -1.60004e-17     -0.261307      -2.48319
    # drift_1..0     -1.60004e-17     -0.261307      -2.48319
    # drift_1..1     -1.14252e-17     -0.186588      -1.98881
    # mb_entry       -6.90158e-18     -0.112711          -1.5
    # mb..entry_map  -6.90158e-18     -0.112711          -1.5
    # mb..0          -6.90158e-18     -0.112711          -1.5
    # mb..1          -6.85007e-18      -0.11187      -1.49442
    # mb..2          -3.04763e-18    -0.0497715     -0.998347
    # mb..3           -7.6238e-19    -0.0124506     -0.499793
    # mid                       0             0             0
    # mb..4                     0             0             0
    # mb..5           -7.6238e-19    -0.0124506      0.499793
    # mb..6          -3.04763e-18    -0.0497715      0.998347
    # mb..7          -6.85007e-18      -0.11187       1.49442
    # mb..exit_map   -6.90158e-18     -0.112711           1.5
    # mb_exit        -6.90158e-18     -0.112711           1.5
    # drift_2..0     -6.90158e-18     -0.112711           1.5
    # drift_2..1     -1.14252e-17     -0.186588       1.98881
    # end            -1.60004e-17     -0.261307       2.48319
    # _end_point     -1.60004e-17     -0.261307       2.48319

    xo.assert_allclose(sv_curved['X'], 0, atol=5e-14)
    xo.assert_allclose(sv_curved['Z'], np.array([
        -2.48319461, -2.48319461, -1.98880907, -1.5       , -1.5       ,
        -1.5       , -1.49442329, -0.99834662, -0.49979325,  0.        ,
        0.        ,  0.49979325,  0.99834662,  1.49442329,  1.5       ,
        1.5       ,  1.5       ,  1.98880907,  2.48319461,  2.48319461
    ]), atol=1e-8)
    xo.assert_allclose(sv_curved['Y'], np.array([
        -0.26130674, -0.26130674, -0.18658768, -0.11271141, -0.11271141,
        -0.11271141, -0.11187018, -0.04977152, -0.0124506 ,  0.        ,
        0.        , -0.0124506 , -0.04977152, -0.11187018, -0.11271141,
        -0.11271141, -0.11271141, -0.18658768, -0.26130674, -0.26130674
    ]), atol=1e-8)

    sv_curved.cols['theta phi psi']
    # is:
    # SurveyTable: 20 rows, 4 cols
    # name                  theta           phi           psi
    # start           9.25436e-18          0.15  -6.95382e-19
    # drift_1..0      9.25436e-18          0.15  -6.95382e-19
    # drift_1..1      9.25436e-18          0.15  -6.95382e-19
    # mb_entry        9.25436e-18          0.15  -6.95382e-19
    # mb..entry_map   9.25436e-18          0.15  -6.95382e-19
    # mb..0           9.25436e-18          0.15  -6.95382e-19
    # mb..1           9.21918e-18      0.149438  -6.90133e-19
    # mb..2           6.12056e-18     0.0996254  -3.05134e-19
    # mb..3           3.05267e-18     0.0498127  -7.60467e-20
    # mid                       0             0             0
    # mb..4                     0             0             0
    # mb..5          -3.05267e-18    -0.0498127  -7.60467e-20
    # mb..6          -6.12056e-18    -0.0996254  -3.05134e-19
    # mb..7          -9.21918e-18     -0.149438  -6.90133e-19
    # mb..exit_map   -9.25436e-18         -0.15  -6.95382e-19
    # mb_exit        -9.25436e-18         -0.15  -6.95382e-19
    # drift_2..0     -9.25436e-18         -0.15  -6.95382e-19
    # drift_2..1     -9.25436e-18         -0.15  -6.95382e-19
    # end            -9.25436e-18         -0.15  -6.95382e-19
    # _end_point     -9.25436e-18         -0.15  -6.95382e-19

    xo.assert_allclose(sv_curved['theta'], 0, atol=5e-14)
    xo.assert_allclose(sv_curved['psi'], 0, atol=5e-14)
    xo.assert_allclose(sv_curved['phi'], np.array([
            0.15      ,  0.15      ,  0.15      ,  0.15      ,  0.15      ,
            0.15      ,  0.14943813,  0.09962542,  0.04981271,  0.        ,
            0.        , -0.04981271, -0.09962542, -0.14943813, -0.15      ,
        -0.15      , -0.15      , -0.15      , -0.15      , -0.15      ],
        ), atol=1e-8)

    for nn in ['start', 'end']:
        xo.assert_allclose(sv_straight['X', nn], sv_curved['X', nn], atol=5e-14)
        xo.assert_allclose(sv_straight['Y', nn], sv_curved['Y', nn], atol=5e-14)
        xo.assert_allclose(sv_straight['Z', nn], sv_curved['Z', nn], atol=5e-14)
        xo.assert_allclose(sv_straight['theta', nn], sv_curved['theta', nn], atol=5e-14)
        xo.assert_allclose(sv_straight['phi', nn], sv_curved['phi', nn], atol=5e-14)
        xo.assert_allclose(sv_straight['psi', nn], sv_curved['psi', nn], atol=5e-14)

    xo.assert_allclose(sv_straight_start['X'], sv_straight['X'], atol=5e-14)
    xo.assert_allclose(sv_straight_start['Y'], sv_straight['Y'], atol=5e-14)
    xo.assert_allclose(sv_straight_start['Z'], sv_straight['Z'], atol=5e-14)
    xo.assert_allclose(sv_straight_start['theta'], sv_straight['theta'], atol=5e-14)
    xo.assert_allclose(sv_straight_start['phi'], sv_straight['phi'], atol=5e-14)
    xo.assert_allclose(sv_straight_start['psi'], sv_straight['psi'], atol=5e-14)

    xo.assert_allclose(sv_straight_end['X'], sv_straight['X'], atol=5e-14)
    xo.assert_allclose(sv_straight_end['Y'], sv_straight['Y'], atol=5e-14)
    xo.assert_allclose(sv_straight_end['Z'], sv_straight['Z'], atol=5e-14)
    xo.assert_allclose(sv_straight_end['theta'], sv_straight['theta'], atol=5e-14)
    xo.assert_allclose(sv_straight_end['phi'], sv_straight['phi'], atol=5e-14)
    xo.assert_allclose(sv_straight_end['psi'], sv_straight['psi'], atol=5e-14)

    xo.assert_allclose(sv_curved_start['X'], sv_curved['X'], atol=5e-14)
    xo.assert_allclose(sv_curved_start['Y'], sv_curved['Y'], atol=5e-14)
    xo.assert_allclose(sv_curved_start['Z'], sv_curved['Z'], atol=5e-14)
    xo.assert_allclose(sv_curved_start['theta'], sv_curved['theta'], atol=5e-14)
    xo.assert_allclose(sv_curved_start['phi'], sv_curved['phi'], atol=5e-14)
    xo.assert_allclose(sv_curved_start['psi'], sv_curved['psi'], atol=5e-14)

    xo.assert_allclose(sv_curved_end['X'], sv_curved['X'], atol=5e-14)
    xo.assert_allclose(sv_curved_end['Y'], sv_curved['Y'], atol=5e-14)
    xo.assert_allclose(sv_curved_end['Z'], sv_curved['Z'], atol=5e-14)
    xo.assert_allclose(sv_curved_end['theta'], sv_curved['theta'], atol=5e-14)
    xo.assert_allclose(sv_curved_end['phi'], sv_curved['phi'], atol=5e-14)
    xo.assert_allclose(sv_curved_end['psi'], sv_curved['psi'], atol=5e-14)
    xo.assert_allclose(tw_straight['X', 'mid'], 0, atol=5e-14)
    xo.assert_allclose(tw_straight['Y', 'mid'], 0, atol=5e-14)
    xo.assert_allclose(tw_straight['Z', 'mid'], 0, atol=5e-14)
    xo.assert_allclose(tw_curved['X', 'mid'], 0, atol=5e-14)
    xo.assert_allclose(tw_curved['Y', 'mid'], 0, atol=5e-14)
    xo.assert_allclose(tw_curved['Z', 'mid'], 0, atol=5e-14)
    xo.assert_allclose(tw_straight['X', 'mb_entry'], tw_curved['X', 'mb_entry'], atol=5e-14)
    xo.assert_allclose(tw_straight['Y', 'mb_entry'], tw_curved['Y', 'mb_entry'], atol=5e-14)
    xo.assert_allclose(tw_straight['Z', 'mb_entry'], tw_curved['Z', 'mb_entry'], atol=5e-14)
    xo.assert_allclose(tw_straight['X', 'mb_exit'], tw_curved['X', 'mb_exit'], atol=5e-14)
    xo.assert_allclose(tw_straight['Y', 'mb_exit'], tw_curved['Y', 'mb_exit'], atol=5e-14)
    xo.assert_allclose(tw_straight['Z', 'mb_exit'], tw_curved['Z', 'mb_exit'], atol=5e-14)

    xo.assert_allclose(tw_straight['x', 'mb_entry'], 0, atol=5e-14)
    xo.assert_allclose(tw_straight['y', 'mb_entry'], 0, atol=5e-14)
    xo.assert_allclose(tw_straight['x', 'mb_exit'], 0, atol=5e-14)
    xo.assert_allclose(tw_straight['y', 'mb_exit'], 0, atol=5e-14)
    xo.assert_allclose(tw_curved['x', 'mb_entry'], 0, atol=5e-14)
    xo.assert_allclose(tw_curved['y', 'mb_entry'], 0, atol=5e-14)
    xo.assert_allclose(tw_curved['x', 'mb_exit'], 0, atol=5e-14)
    xo.assert_allclose(tw_curved['y', 'mb_exit'], 0, atol=5e-14)

    xo.assert_allclose(tw_no_slice_curved['x', 'start'], tw_curved['x', 'start'],
                    atol=5e-14)
    xo.assert_allclose(tw_no_slice_curved['y', 'start'], tw_curved['y', 'start'],
                    atol=5e-14)
    xo.assert_allclose(tw_no_slice_curved['x', 'end'], tw_curved['x', 'end'],
                    atol=5e-14)
    xo.assert_allclose(tw_no_slice_curved['y', 'end'], tw_curved['y', 'end'],
                    atol=5e-14)
    xo.assert_allclose(tw_no_slice_straight['x', 'start'], tw_curved['x', 'start'],
                        atol=5e-14)
    xo.assert_allclose(tw_no_slice_straight['y', 'start'], tw_curved['y', 'start'],
                        atol=5e-14)
    xo.assert_allclose(tw_no_slice_straight['x', 'end'], tw_curved['x', 'end'],
                        atol=5e-14)
    xo.assert_allclose(tw_no_slice_straight['y', 'end'], tw_curved['y', 'end'],
                        atol=5e-14)

    for nn in ['start', 'end']:
        # Compare no_slice survey vs curved survey
        xo.assert_allclose(sv_no_slice_curved_start['X', nn], sv_curved['X', nn], atol=5e-14)
        xo.assert_allclose(sv_no_slice_curved_start['Y', nn], sv_curved['Y', nn], atol=5e-14)
        xo.assert_allclose(sv_no_slice_curved_start['Z', nn], sv_curved['Z', nn], atol=5e-14)
        xo.assert_allclose(sv_no_slice_curved_start['theta', nn], sv_curved['theta', nn], atol=5e-14)
        xo.assert_allclose(sv_no_slice_curved_start['phi', nn], sv_curved['phi', nn], atol=5e-14)
        xo.assert_allclose(sv_no_slice_curved_start['psi', nn], sv_curved['psi', nn], atol=5e-14)
        xo.assert_allclose(sv_no_slice_curved_start['s', nn], sv_curved['s', nn], atol=5e-14)

        xo.assert_allclose(sv_no_slice_curved_end['X', nn], sv_curved['X', nn], atol=5e-14)
        xo.assert_allclose(sv_no_slice_curved_end['Y', nn], sv_curved['Y', nn], atol=5e-14)
        xo.assert_allclose(sv_no_slice_curved_end['Z', nn], sv_curved['Z', nn], atol=5e-14)
        xo.assert_allclose(sv_no_slice_curved_end['theta', nn], sv_curved['theta', nn], atol=5e-14)
        xo.assert_allclose(sv_no_slice_curved_end['phi', nn], sv_curved['phi', nn], atol=5e-14)
        xo.assert_allclose(sv_no_slice_curved_end['psi', nn], sv_curved['psi', nn], atol=5e-14)
        xo.assert_allclose(sv_no_slice_curved_end['s', nn], sv_curved['s', nn], atol=5e-14)

    for nn in ['start', 'end']:
        xo.assert_allclose(sv_no_slice_straight_start['X', nn], sv_straight['X', nn], atol=5e-14)
        xo.assert_allclose(sv_no_slice_straight_start['Y', nn], sv_straight['Y', nn], atol=5e-14)
        xo.assert_allclose(sv_no_slice_straight_start['Z', nn], sv_straight['Z', nn], atol=5e-14)
        xo.assert_allclose(sv_no_slice_straight_start['theta', nn], sv_straight['theta', nn], atol=5e-14)
        xo.assert_allclose(sv_no_slice_straight_start['phi', nn], sv_straight['phi', nn], atol=5e-14)
        xo.assert_allclose(sv_no_slice_straight_start['psi', nn], sv_straight['psi', nn], atol=5e-14)
        xo.assert_allclose(sv_no_slice_straight_start['s', nn], sv_straight['s', nn], atol=5e-14)

        xo.assert_allclose(sv_no_slice_straight_end['X', nn], sv_straight['X', nn], atol=5e-14)
        xo.assert_allclose(sv_no_slice_straight_end['Y', nn], sv_straight['Y', nn], atol=5e-14)
        xo.assert_allclose(sv_no_slice_straight_end['Z', nn], sv_straight['Z', nn], atol=5e-14)
        xo.assert_allclose(sv_no_slice_straight_end['theta', nn], sv_straight['theta', nn], atol=5e-14)
        xo.assert_allclose(sv_no_slice_straight_end['phi', nn], sv_straight['phi', nn], atol=5e-14)
        xo.assert_allclose(sv_no_slice_straight_end['psi', nn], sv_straight['psi', nn], atol=5e-14)
        xo.assert_allclose(sv_no_slice_straight_end['s', nn], sv_straight['s', nn], atol=5e-14)

def test_rbend_straight_body_thin_slices_coarse():

    env = xt.Environment(particle_ref=xt.Particles(p0c=10e9))

    edge_model = 'full'

    line = env.new_line(length=5, components=[
        env.new('mb', 'RBend', angle=0.3, k0_from_h=True, length_straight=3,
                rot_s_rad=np.pi/2,
                model='bend-kick-bend',
                rbend_model='straight-body', edge_entry_model=edge_model, edge_exit_model=edge_model,
                at=2.5)])
    line.insert('start', xt.Marker(), at=0)
    line.append('end', xt.Marker())

    line_no_slice = line.copy(shallow=True)

    line.slice_thick_elements(
            slicing_strategies=[
                # Slicing with thin elements
                xt.Strategy(slicing=None),
                xt.Strategy(slicing=xt.Teapot(10), element_type=xt.RBend),
            ])

    line.insert('mid', xt.Marker(), at=2.5)

    line['mb'].rbend_model = 'straight-body'
    sv_straight = line.survey(element0='mid', Y0=-line['mb'].sagitta/2)
    tt_straight = line.get_table(attr=True)
    tw_straight = line.twiss(betx=1, bety=1)
    p_straight = (sv_straight.p0 + tw_straight.x[:, None] * sv_straight['ex']
                                + tw_straight.y[:, None] * sv_straight['ey'])
    tw_straight['X'] = p_straight[:, 0]
    tw_straight['Y'] = p_straight[:, 1]
    tw_straight['Z'] = p_straight[:, 2]

    line['mb'].rbend_model = 'curved-body'
    sv_curved = line.survey(element0='mid')
    tt_curved = line.get_table(attr=True)
    tw_curved = line.twiss(betx=1, bety=1)
    p_curved = (sv_curved.p0 + tw_curved.x[:, None] * sv_curved['ex']
                            + tw_curved.y[:, None] * sv_curved['ey'])
    tw_curved['X'] = p_curved[:, 0]
    tw_curved['Y'] = p_curved[:, 1]
    tw_curved['Z'] = p_curved[:, 2]

    assert np.all(sv_straight['name'] == [
        'start', '||drift_1', 'mb_entry', 'mb..entry_map', 'drift_mb..0',
       'mb..0', 'drift_mb..1', 'mb..1', 'drift_mb..2', 'mb..2',
       'drift_mb..3', 'mb..3', 'drift_mb..4', 'mb..4', 'drift_mb..5..0',
       'mid', 'drift_mb..5..1', 'mb..5', 'drift_mb..6', 'mb..6',
       'drift_mb..7', 'mb..7', 'drift_mb..8', 'mb..8', 'drift_mb..9',
       'mb..9', 'drift_mb..10', 'mb..exit_map', 'mb_exit', '||drift_2',
       'end', '_end_point'])

    # Assert entire columns using np.all
    assert np.all(sv_straight['element_type'] == [
        'Marker', 'Drift', 'Marker', 'ThinSliceRBendEntry',
        'DriftSliceRBend', 'ThinSliceRBend', 'DriftSliceRBend',
        'ThinSliceRBend', 'DriftSliceRBend', 'ThinSliceRBend',
        'DriftSliceRBend', 'ThinSliceRBend', 'DriftSliceRBend',
        'ThinSliceRBend', 'DriftSliceRBend', 'Marker', 'DriftSliceRBend',
        'ThinSliceRBend', 'DriftSliceRBend', 'ThinSliceRBend',
        'DriftSliceRBend', 'ThinSliceRBend', 'DriftSliceRBend',
        'ThinSliceRBend', 'DriftSliceRBend', 'ThinSliceRBend',
        'DriftSliceRBend', 'ThinSliceRBendExit', 'Marker', 'Drift',
        'Marker', ''])

    xo.assert_allclose(
        sv_straight['angle'], np.array([
        0.  , 0.  , 0.  , 0.15, 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
        0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
        0.  , 0.  , 0.  , 0.  , 0.  , 0.15, 0.  , 0.  , 0.  , 0.  ]),
        atol=1e-12
    )

    xo.assert_allclose(sv_straight['s'], np.array([
        0.        , 0.        , 0.9943602 , 0.9943602 , 0.9943602 ,
        1.13123654, 1.13123654, 1.4354062 , 1.4354062 , 1.73957586,
        1.73957586, 2.04374551, 2.04374551, 2.34791517, 2.34791517,
        2.5       , 2.5       , 2.65208483, 2.65208483, 2.95625449,
        2.95625449, 3.26042414, 3.26042414, 3.5645938 , 3.5645938 ,
        3.86876346, 3.86876346, 4.0056398 , 4.0056398 , 4.0056398 ,
        5.        , 5.        ]
    ), atol=1e-5)

    xo.assert_allclose(
        sv_straight['rot_s_rad'],
        np.array([0.        , 0.        , 0.        , 1.57079633, 0.        ,
        1.57079633, 0.        , 1.57079633, 0.        , 1.57079633,
        0.        , 1.57079633, 0.        , 1.57079633, 0.        ,
        0.        , 0.        , 1.57079633, 0.        , 1.57079633,
        0.        , 1.57079633, 0.        , 1.57079633, 0.        ,
        1.57079633, 0.        , 1.57079633, 0.        , 0.        ,
        0.        , 0.        ]),
        atol=1e-8)


    xo.assert_allclose(sv_straight['X'], 0, atol=5e-14)
    xo.assert_allclose(sv_straight['Z'], np.array([
        -2.48319461, -2.48319461, -1.5       , -1.5       , -1.5       ,
        -1.36363636, -1.36363636, -1.06060606, -1.06060606, -0.75757576,
        -0.75757576, -0.45454545, -0.45454545, -0.15151515, -0.15151515,
            0.        ,  0.        ,  0.15151515,  0.15151515,  0.45454545,
            0.45454545,  0.75757576,  0.75757576,  1.06060606,  1.06060606,
            1.36363636,  1.36363636,  1.5       ,  1.5       ,  1.5       ,
            2.48319461,  2.48319461]),
            atol=1e-8)
    xo.assert_allclose(sv_straight['Y'], np.array(
        [-0.26130674, -0.26130674, -0.11271141, -0.11271141, -0.05635571,
        -0.05635571, -0.05635571, -0.05635571, -0.05635571, -0.05635571,
        -0.05635571, -0.05635571, -0.05635571, -0.05635571, -0.05635571,
        -0.05635571, -0.05635571, -0.05635571, -0.05635571, -0.05635571,
        -0.05635571, -0.05635571, -0.05635571, -0.05635571, -0.05635571,
        -0.05635571, -0.05635571, -0.05635571, -0.11271141, -0.11271141,
        -0.26130674, -0.26130674]),
        atol=1e-8)


    xo.assert_allclose(sv_straight['theta'], 0, atol=5e-14)
    xo.assert_allclose(sv_straight['psi'], 0, atol=5e-14)
    xo.assert_allclose(sv_straight['phi'], np.array([
            0.15,  0.15,  0.15,  0.15,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
            0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
            0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
            0.  , -0.15, -0.15, -0.15, -0.15]))

    assert np.all(sv_curved['name'] == sv_straight['name'])
    assert np.all(sv_curved['element_type'] == sv_straight['element_type'])

    xo.assert_allclose(
        sv_curved['angle'],
        np.array([
        0.  , 0.  , 0.  , 0.  , 0.  , 0.03, 0.  , 0.03, 0.  , 0.03, 0.  ,
        0.03, 0.  , 0.03, 0.  , 0.  , 0.  , 0.03, 0.  , 0.03, 0.  , 0.03,
        0.  , 0.03, 0.  , 0.03, 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ]),
        atol=1e-8
    )

    xo.assert_allclose(sv_curved['s'], sv_straight['s'], atol=1e-12)
    xo.assert_allclose(sv_curved['rot_s_rad'], sv_straight['rot_s_rad'], atol=1e-12)


    xo.assert_allclose(sv_curved['X'], 0, atol=5e-14)
    xo.assert_allclose(sv_curved['Z'], np.array([
        -2.48319478, -2.48319478, -1.50000017, -1.50000017, -1.50000017,
        -1.3646608 , -1.3646608 , -1.06267854, -1.06267854, -0.75973993,
        -0.75973993, -0.45611762, -0.45611762, -0.15208483, -0.15208483,
            0.        ,  0.        ,  0.15208483,  0.15208483,  0.45611762,
            0.45611762,  0.75973993,  0.75973993,  1.06267854,  1.06267854,
            1.3646608 ,  1.3646608 ,  1.50000017,  1.50000017,  1.50000017,
            2.48319478,  2.48319478]), atol=1e-8)
    xo.assert_allclose(sv_curved['Y'], np.array(
        [-0.26016398, -0.26016398, -0.11156865, -0.11156865, -0.11156865,
        -0.0911141 , -0.0911141 , -0.05470128, -0.05470128, -0.02736295,
        -0.02736295, -0.00912372, -0.00912372,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        , -0.00912372,
        -0.00912372, -0.02736295, -0.02736295, -0.05470128, -0.05470128,
        -0.0911141 , -0.0911141 , -0.11156865, -0.11156865, -0.11156865,
        -0.26016398, -0.26016398]), atol=1e-8)

    xo.assert_allclose(sv_curved['theta'], 0, atol=5e-14)
    xo.assert_allclose(sv_curved['psi'], 0, atol=5e-14)
    xo.assert_allclose(sv_curved['phi'], np.array([
            0.15,  0.15,  0.15,  0.15,  0.15,  0.15,  0.12,  0.12,  0.09,
            0.09,  0.06,  0.06,  0.03,  0.03,  0.  ,  0.  ,  0.  ,  0.  ,
        -0.03, -0.03, -0.06, -0.06, -0.09, -0.09, -0.12, -0.12, -0.15,
        -0.15, -0.15, -0.15, -0.15, -0.15],
        ), atol=1e-8)

def test_rbend_straight_body_thin_slices_fine():

    env = xt.Environment(particle_ref=xt.Particles(p0c=10e9))

    edge_model = 'full'

    line = env.new_line(length=5, components=[
        env.new('mb', 'RBend', angle=0.3, k0_from_h=True, length_straight=3,
                rot_s_rad=np.pi/2,
                model='bend-kick-bend',
                rbend_model='straight-body',
                edge_entry_model=edge_model, edge_exit_model=edge_model,
                at=2.5)])
    line.insert('start', xt.Marker(), at=0)
    line.append('end', xt.Marker())
    line.configure_drift_model('exact')

    line_no_slice = line.copy(shallow=True)

    line.slice_thick_elements(
            slicing_strategies=[
                # Slicing with thin elements
                xt.Strategy(slicing=None),
                xt.Strategy(slicing=xt.Teapot(1000), element_type=xt.RBend),
            ])

    line.insert('mid', xt.Marker(), at=2.5)

    line['mb'].rbend_model = 'straight-body'
    sv_straight = line.survey(element0='mid', Y0=-line['mb'].sagitta/2)
    tt_straight = line.get_table(attr=True)
    tw_straight = line.twiss(betx=1, bety=1)
    p_straight = (sv_straight.p0 + tw_straight.x[:, None] * sv_straight['ex']
                                + tw_straight.y[:, None] * sv_straight['ey'])
    tw_straight['X'] = p_straight[:, 0]
    tw_straight['Y'] = p_straight[:, 1]
    tw_straight['Z'] = p_straight[:, 2]

    sv_straight_start = line.survey(element0='start',
                                    X0=sv_straight['X', 'start'],
                                    Y0=sv_straight['Y', 'start'],
                                    Z0=sv_straight['Z', 'start'],
                                    theta0=sv_straight['theta', 'start'],
                                    phi0=sv_straight['phi', 'start'],
                                    psi0=sv_straight['psi', 'start'])
    sv_straight_end = line.survey(element0='end',
                                    X0=sv_straight['X', 'end'],
                                    Y0=sv_straight['Y', 'end'],
                                    Z0=sv_straight['Z', 'end'],
                                    theta0=sv_straight['theta', 'end'],
                                    phi0=sv_straight['phi', 'end'],
                                    psi0=sv_straight['psi', 'end'])

    sv_no_slice_straight_start = line_no_slice.survey(element0='start',
                                    X0=sv_straight['X', 'start'],
                                    Y0=sv_straight['Y', 'start'],
                                    Z0=sv_straight['Z', 'start'],
                                    theta0=sv_straight['theta', 'start'],
                                    phi0=sv_straight['phi', 'start'],
                                    psi0=sv_straight['psi', 'start'])
    sv_no_slice_straight_end = line_no_slice.survey(element0='end',
                                    X0=sv_straight['X', 'end'],
                                    Y0=sv_straight['Y', 'end'],
                                    Z0=sv_straight['Z', 'end'],
                                    theta0=sv_straight['theta', 'end'],
                                    phi0=sv_straight['phi', 'end'],
                                    psi0=sv_straight['psi', 'end'])
    tw_no_slice_straight = line_no_slice.twiss(betx=1, bety=1)

    line['mb'].rbend_model = 'curved-body'
    sv_curved = line.survey(element0='mid')
    tt_curved = line.get_table(attr=True)
    tw_curved = line.twiss(betx=1, bety=1)
    p_curved = (sv_curved.p0 + tw_curved.x[:, None] * sv_curved['ex']
                            + tw_curved.y[:, None] * sv_curved['ey'])
    tw_curved['X'] = p_curved[:, 0]
    tw_curved['Y'] = p_curved[:, 1]
    tw_curved['Z'] = p_curved[:, 2]

    sv_curved_start = line.survey(element0='start',
                                    X0=sv_curved['X', 'start'],
                                    Y0=sv_curved['Y', 'start'],
                                    Z0=sv_curved['Z', 'start'],
                                    theta0=sv_curved['theta', 'start'],
                                    phi0=sv_curved['phi', 'start'],
                                    psi0=sv_curved['psi', 'start'])
    sv_curved_end = line.survey(element0='end',
                                    X0=sv_curved['X', 'end'],
                                    Y0=sv_curved['Y', 'end'],
                                    Z0=sv_curved['Z', 'end'],
                                    theta0=sv_curved['theta', 'end'],
                                    phi0=sv_curved['phi', 'end'],
                                    psi0=sv_curved['psi', 'end'])
    sv_no_slice_curved_start = line_no_slice.survey(element0='start',
                                    X0=sv_curved['X', 'start'],
                                    Y0=sv_curved['Y', 'start'],
                                    Z0=sv_curved['Z', 'start'],
                                    theta0=sv_curved['theta', 'start'],
                                    phi0=sv_curved['phi', 'start'],
                                    psi0=sv_curved['psi', 'start'])
    sv_no_slice_curved_end = line_no_slice.survey(element0='end',
                                    X0=sv_curved['X', 'end'],
                                    Y0=sv_curved['Y', 'end'],
                                    Z0=sv_curved['Z', 'end'],
                                    theta0=sv_curved['theta', 'end'],
                                    phi0=sv_curved['phi', 'end'],
                                    psi0=sv_curved['psi', 'end'])
    tw_no_slice_curved = line_no_slice.twiss(betx=1, bety=1)


    for nn in ['start', 'end']:
        xo.assert_allclose(sv_straight['X', nn], sv_curved['X', nn], atol=5e-14)
        xo.assert_allclose(sv_straight['Y', nn], sv_curved['Y', nn], atol=2e-7)
        xo.assert_allclose(sv_straight['Z', nn], sv_curved['Z', nn], atol=2e-7)
        xo.assert_allclose(sv_straight['theta', nn], sv_curved['theta', nn], atol=5e-14)
        xo.assert_allclose(sv_straight['phi', nn], sv_curved['phi', nn], atol=5e-14)
        xo.assert_allclose(sv_straight['psi', nn], sv_curved['psi', nn], atol=5e-14)

    xo.assert_allclose(sv_straight_start['X'], sv_straight['X'], atol=5e-14)
    xo.assert_allclose(sv_straight_start['Y'], sv_straight['Y'], atol=2e-7)
    xo.assert_allclose(sv_straight_start['Z'], sv_straight['Z'], atol=5e-14)
    xo.assert_allclose(sv_straight_start['theta'], sv_straight['theta'], atol=5e-14)
    xo.assert_allclose(sv_straight_start['phi'], sv_straight['phi'], atol=5e-14)
    xo.assert_allclose(sv_straight_start['psi'], sv_straight['psi'], atol=5e-14)

    xo.assert_allclose(sv_straight_end['X'], sv_straight['X'], atol=5e-14)
    xo.assert_allclose(sv_straight_end['Y'], sv_straight['Y'], atol=5e-14)
    xo.assert_allclose(sv_straight_end['Z'], sv_straight['Z'], atol=5e-14)
    xo.assert_allclose(sv_straight_end['theta'], sv_straight['theta'], atol=5e-14)
    xo.assert_allclose(sv_straight_end['phi'], sv_straight['phi'], atol=5e-14)
    xo.assert_allclose(sv_straight_end['psi'], sv_straight['psi'], atol=5e-14)

    xo.assert_allclose(sv_curved_start['X'], sv_curved['X'], atol=5e-14)
    xo.assert_allclose(sv_curved_start['Y'], sv_curved['Y'], atol=2e-7)
    xo.assert_allclose(sv_curved_start['Z'], sv_curved['Z'], atol=2e-7)
    xo.assert_allclose(sv_curved_start['theta'], sv_curved['theta'], atol=5e-14)
    xo.assert_allclose(sv_curved_start['phi'], sv_curved['phi'], atol=5e-14)
    xo.assert_allclose(sv_curved_start['psi'], sv_curved['psi'], atol=5e-14)

    xo.assert_allclose(sv_curved_end['X'], sv_curved['X'], atol=5e-14)
    xo.assert_allclose(sv_curved_end['Y'], sv_curved['Y'], atol=1e-12)
    xo.assert_allclose(sv_curved_end['Z'], sv_curved['Z'], atol=1e-12)
    xo.assert_allclose(sv_curved_end['theta'], sv_curved['theta'], atol=5e-14)
    xo.assert_allclose(sv_curved_end['phi'], sv_curved['phi'], atol=1e-12)
    xo.assert_allclose(sv_curved_end['psi'], sv_curved['psi'], atol=1e-12)
    xo.assert_allclose(tw_straight['X', 'mid'], 0, atol=5e-14)
    xo.assert_allclose(tw_straight['Y', 'mid'], 0, atol=1e-3) # I am truncating the hamiltonian
    xo.assert_allclose(tw_straight['Z', 'mid'], 0, atol=1e-12)
    xo.assert_allclose(tw_curved['X', 'mid'], 0, atol=5e-14)
    xo.assert_allclose(tw_curved['Y', 'mid'], 0,  atol=1e-12)
    xo.assert_allclose(tw_curved['Z', 'mid'], 0, atol=1e-12)
    xo.assert_allclose(tw_straight['X', 'mb_entry'], tw_curved['X', 'mb_entry'], atol=5e-14)
    xo.assert_allclose(tw_straight['Y', 'mb_entry'], tw_curved['Y', 'mb_entry'], atol=2e-7)
    xo.assert_allclose(tw_straight['Z', 'mb_entry'], tw_curved['Z', 'mb_entry'], atol=2e-7)
    xo.assert_allclose(tw_straight['X', 'mb_exit'], tw_curved['X', 'mb_exit'], atol=5e-14)
    xo.assert_allclose(tw_straight['Y', 'mb_exit'], tw_curved['Y', 'mb_exit'], atol=2e-7)
    xo.assert_allclose(tw_straight['Z', 'mb_exit'], tw_curved['Z', 'mb_exit'], atol=2e-7)

    xo.assert_allclose(tw_straight['x', 'mb_entry'], 0, atol=1e-13)
    xo.assert_allclose(tw_straight['y', 'mb_entry'], 0, atol=1e-13)
    xo.assert_allclose(tw_straight['x', 'mb_exit'], 0, atol=1e-13)
    xo.assert_allclose(tw_straight['y', 'mb_exit'], 0, atol=1e-13)
    xo.assert_allclose(tw_curved['x', 'mb_entry'], 0, atol=1e-13)
    xo.assert_allclose(tw_curved['y', 'mb_entry'], 0, atol=1e-13)
    xo.assert_allclose(tw_curved['x', 'mb_exit'], 0, atol=1e-13)
    xo.assert_allclose(tw_curved['y', 'mb_exit'], 0, atol=1e-13)

    xo.assert_allclose(tw_no_slice_curved['x', 'start'], tw_curved['x', 'start'],
                    atol=5e-14)
    xo.assert_allclose(tw_no_slice_curved['y', 'start'], tw_curved['y', 'start'],
                    atol=5e-14)
    xo.assert_allclose(tw_no_slice_curved['x', 'end'], tw_curved['x', 'end'],
                    atol=5e-14)
    xo.assert_allclose(tw_no_slice_curved['y', 'end'], tw_curved['y', 'end'],
                    atol=5e-14)
    xo.assert_allclose(tw_no_slice_straight['x', 'start'], tw_curved['x', 'start'],
                        atol=5e-14)
    xo.assert_allclose(tw_no_slice_straight['y', 'start'], tw_curved['y', 'start'],
                        atol=5e-14)
    xo.assert_allclose(tw_no_slice_straight['x', 'end'], tw_curved['x', 'end'],
                        atol=5e-14)
    xo.assert_allclose(tw_no_slice_straight['y', 'end'], tw_curved['y', 'end'],
                        atol=5e-14)

    for nn in ['start', 'end']:
        # Compare no_slice survey vs curved survey
        xo.assert_allclose(sv_no_slice_curved_start['X', nn], sv_curved['X', nn], atol=5e-11)
        xo.assert_allclose(sv_no_slice_curved_start['Y', nn], sv_curved['Y', nn], atol=5e-11)
        xo.assert_allclose(sv_no_slice_curved_start['Z', nn], sv_curved['Z', nn], atol=5e-11)
        xo.assert_allclose(sv_no_slice_curved_start['theta', nn], sv_curved['theta', nn], atol=5e-11)
        xo.assert_allclose(sv_no_slice_curved_start['phi', nn], sv_curved['phi', nn], atol=5e-11)
        xo.assert_allclose(sv_no_slice_curved_start['psi', nn], sv_curved['psi', nn], atol=5e-11)
        xo.assert_allclose(sv_no_slice_curved_start['s', nn], sv_curved['s', nn], atol=5e-11)

        xo.assert_allclose(sv_no_slice_curved_end['X', nn], sv_curved['X', nn], atol=5e-11)
        xo.assert_allclose(sv_no_slice_curved_end['Y', nn], sv_curved['Y', nn], atol=5e-11)
        xo.assert_allclose(sv_no_slice_curved_end['Z', nn], sv_curved['Z', nn], atol=5e-11)
        xo.assert_allclose(sv_no_slice_curved_end['theta', nn], sv_curved['theta', nn], atol=5e-11)
        xo.assert_allclose(sv_no_slice_curved_end['phi', nn], sv_curved['phi', nn], atol=5e-11)
        xo.assert_allclose(sv_no_slice_curved_end['psi', nn], sv_curved['psi', nn], atol=5e-11)
        xo.assert_allclose(sv_no_slice_curved_end['s', nn], sv_curved['s', nn], atol=5e-11)

    for nn in ['start', 'end']:
        # Compare no_slice survey vs straight survey
        xo.assert_allclose(sv_no_slice_straight_start['X', nn], sv_straight['X', nn], atol=5e-11)
        xo.assert_allclose(sv_no_slice_straight_start['Y', nn], sv_straight['Y', nn], atol=5e-11)
        xo.assert_allclose(sv_no_slice_straight_start['Z', nn], sv_straight['Z', nn], atol=5e-11)
        xo.assert_allclose(sv_no_slice_straight_start['theta', nn], sv_straight['theta', nn], atol=5e-11)
        xo.assert_allclose(sv_no_slice_straight_start['phi', nn], sv_straight['phi', nn], atol=5e-11)
        xo.assert_allclose(sv_no_slice_straight_start['psi', nn], sv_straight['psi', nn], atol=5e-11)
        xo.assert_allclose(sv_no_slice_straight_start['s', nn], sv_straight['s', nn], atol=5e-11)

        xo.assert_allclose(sv_no_slice_straight_end['X', nn], sv_straight['X', nn], atol=5e-11)
        xo.assert_allclose(sv_no_slice_straight_end['Y', nn], sv_straight['Y', nn], atol=5e-11)
        xo.assert_allclose(sv_no_slice_straight_end['Z', nn], sv_straight['Z', nn], atol=5e-11)
        xo.assert_allclose(sv_no_slice_straight_end['theta', nn], sv_straight['theta', nn], atol=5e-11)
        xo.assert_allclose(sv_no_slice_straight_end['phi', nn], sv_straight['phi', nn], atol=5e-11)
        xo.assert_allclose(sv_no_slice_straight_end['psi', nn], sv_straight['psi', nn], atol=5e-11)
        xo.assert_allclose(sv_no_slice_straight_end['s', nn], sv_straight['s', nn], atol=5e-11)

def test_rbend_straight_body_shift():

    env = xt.Environment(particle_ref=xt.Particles(p0c=10e9))

    edge_model = 'linear'
    shift = 0.3  # m

    line = env.new_line(length=5, components=[
        env.new('mb', 'RBend', angle=0.3, k0_from_h=True, length_straight=3,
                model='bend-kick-bend',
                rbend_model='straight-body',
                rbend_shift=shift,
                edge_entry_model=edge_model, edge_exit_model=edge_model,
                at=2.5)])
    line.insert('start', xt.Marker(), at=0)
    line.append('end', xt.Marker())
    line.cut_at_s(np.linspace(0, line.get_length(), 11))
    line.insert('mid', xt.Marker(), at=2.5)

    tw = line.twiss(betx=1, bety=1)
    sv = line.survey(element0='mid')

    xo.assert_allclose(tw['x', 'mb_entry'], 0, atol=5e-14)
    xo.assert_allclose(tw['x', 'mid'], line['mb'].sagitta / 2 - shift, atol=5e-14)
    xo.assert_allclose(tw['x', 'mb..0'], -line['mb'].sagitta / 2 - shift, atol=5e-14)
    xo.assert_allclose(tw['x', 'mb..exit_map'], -line['mb'].sagitta / 2 - shift, atol=5e-14)
    xo.assert_allclose(tw['x', 'mb_exit'], 0, atol=5e-14)
    xo.assert_allclose(tw.y, 0, atol=5e-14)

    xo.assert_allclose(sv.rows['mb_entry':'mb_exit'].X[2:-1], 0, atol=5e-14)
    xo.assert_allclose(sv['X', 'mb_entry'], -line['mb'].sagitta / 2 - shift, atol=5e-14)
    xo.assert_allclose(sv['X', 'mb_exit'], -line['mb'].sagitta / 2 - shift, atol=5e-14)
    xo.assert_allclose(sv.Y, 0, atol=5e-14)

    xo.assert_allclose(sv['theta', 'mb_entry'], 0.15, atol=5e-14)
    xo.assert_allclose(sv['theta', 'mb_exit'], -0.15, atol=5e-14)
    xo.assert_allclose(sv['theta', 'mid'], 0, atol=5e-14)
    xo.assert_allclose(sv.phi, 0, atol=5e-14)
    xo.assert_allclose(sv.psi, 0, atol=5e-14)

    sv_init_start = line.survey(element0='start',
                                X0=sv['X', 'start'],
                                Y0=sv['Y', 'start'],
                                Z0=sv['Z', 'start'],
                                phi0=sv['phi', 'start'],
                                psi0=sv['psi', 'start'],
                                theta0=sv['theta', 'start'])
    sv_iinit_end = line.survey(element0='end',
                            X0=sv['X', 'end'],
                            Y0=sv['Y', 'end'],
                            Z0=sv['Z', 'end'],
                            phi0=sv['phi', 'end'],
                            psi0=sv['psi', 'end'],
                            theta0=sv['theta', 'end'])

    for sv_test in [sv, sv_init_start, sv_iinit_end]:
        xo.assert_allclose(sv_test.X, sv.X, atol=5e-14)
        xo.assert_allclose(sv_test.Y, sv.Y, atol=5e-14)
        xo.assert_allclose(sv_test.Z, sv.Z, atol=5e-14)
        xo.assert_allclose(sv_test.phi, sv.phi, atol=5e-14)
        xo.assert_allclose(sv_test.psi, sv.psi, atol=5e-14)
        xo.assert_allclose(sv_test.theta, sv.theta, atol=5e-14)

    tw_back = line.twiss(init=tw, init_at='end')

    xo.assert_allclose(tw_back.x, tw.x, atol=5e-14)
    xo.assert_allclose(tw_back.y, tw.y, atol=5e-14)
    xo.assert_allclose(tw_back.s, tw.s, atol=5e-14)
    xo.assert_allclose(tw_back.zeta, tw.zeta, atol=5e-14)

    line['mb'].rbend_model = 'curved-body'
    tw_curved = line.twiss(betx=1, bety=1)
    sv_curved = line.survey(element0='mid')

    xo.assert_allclose(tw_curved.x, 0, atol=5e-14)
    xo.assert_allclose(tw_curved.y, 0, atol=5e-14)
    xo.assert_allclose(sv_curved['X', 'mb_entry'], -line['mb'].sagitta, atol=5e-14)
    xo.assert_allclose(sv_curved['X', 'mid'], 0, atol=5e-14)
    xo.assert_allclose(sv_curved['X', 'mb_exit'], -line['mb'].sagitta, atol=5e-14)

    sv_curved_init_start = line.survey(element0='start',
                                    X0=sv_curved['X', 'start'],
                                    Y0=sv_curved['Y', 'start'],
                                    Z0=sv_curved['Z', 'start'],
                                    phi0=sv_curved['phi', 'start'],
                                    psi0=sv_curved['psi', 'start'],
                                    theta0=sv_curved['theta', 'start'])
    sv_curved_init_end = line.survey(element0='end',
                                    X0=sv_curved['X', 'end'],
                                    Y0=sv_curved['Y', 'end'],
                                    Z0=sv_curved['Z', 'end'],
                                    phi0=sv_curved['phi', 'end'],
                                    psi0=sv_curved['psi', 'end'],
                                    theta0=sv_curved['theta', 'end'])

    for sv_test in [sv_curved_init_start, sv_curved_init_end]:
        xo.assert_allclose(sv_test.X, sv_curved.X, atol=5e-14)
        xo.assert_allclose(sv_test.Y, sv_curved.Y, atol=5e-14)
        xo.assert_allclose(sv_test.Z, sv_curved.Z, atol=5e-14)
        xo.assert_allclose(sv_test.phi, sv_curved.phi, atol=5e-14)
        xo.assert_allclose(sv_test.psi, sv_curved.psi, atol=5e-14)
        xo.assert_allclose(sv_test.theta, sv_curved.theta, atol=5e-14)

def test_rbend_straight_body_v_shift():

    env = xt.Environment(particle_ref=xt.Particles(p0c=10e9))

    edge_model = 'linear'
    shift = 0.3  # m

    line = env.new_line(length=5, components=[
        env.new('mb', 'RBend', angle=0.3, k0_from_h=True, length_straight=3,
                model='bend-kick-bend',
                rbend_model='straight-body',
                rot_s_rad=np.pi/2,  # rad
                rbend_shift=shift,
                edge_entry_model=edge_model, edge_exit_model=edge_model,
                at=2.5)])
    line.insert('start', xt.Marker(), at=0)
    line.append('end', xt.Marker())
    line.cut_at_s(np.linspace(0, line.get_length(), 11))
    line.insert('mid', xt.Marker(), at=2.5)

    tw = line.twiss(betx=1, bety=1)
    sv = line.survey(element0='mid')

    xo.assert_allclose(tw['y', 'mb_entry'], 0, atol=5e-14)
    xo.assert_allclose(tw['y', 'mid'], line['mb'].sagitta / 2 - shift, atol=5e-14)
    xo.assert_allclose(tw['y', 'mb..0'], -line['mb'].sagitta / 2 - shift, atol=5e-14)
    xo.assert_allclose(tw['y', 'mb..exit_map'], -line['mb'].sagitta / 2 - shift, atol=5e-14)
    xo.assert_allclose(tw['y', 'mb_exit'], 0, atol=5e-14)
    xo.assert_allclose(tw.x, 0, atol=5e-14)

    xo.assert_allclose(sv.rows['mb_entry':'mb_exit'].Y[2:-1], 0, atol=5e-14)
    xo.assert_allclose(sv['Y', 'mb_entry'], -line['mb'].sagitta / 2 - shift, atol=5e-14)
    xo.assert_allclose(sv['Y', 'mb_exit'], -line['mb'].sagitta / 2 - shift, atol=5e-14)
    xo.assert_allclose(sv.X, 0, atol=5e-14)

    xo.assert_allclose(sv['phi', 'mb_entry'], 0.15, atol=5e-14)
    xo.assert_allclose(sv['phi', 'mb_exit'], -0.15, atol=5e-14)
    xo.assert_allclose(sv['phi', 'mid'], 0, atol=5e-14)
    xo.assert_allclose(sv.theta, 0, atol=5e-14)
    xo.assert_allclose(sv.psi, 0, atol=5e-14)

    sv_init_start = line.survey(element0='start',
                                X0=sv['X', 'start'],
                                Y0=sv['Y', 'start'],
                                Z0=sv['Z', 'start'],
                                phi0=sv['phi', 'start'],
                                psi0=sv['psi', 'start'],
                                theta0=sv['theta', 'start'])
    sv_iinit_end = line.survey(element0='end',
                            X0=sv['X', 'end'],
                            Y0=sv['Y', 'end'],
                            Z0=sv['Z', 'end'],
                            phi0=sv['phi', 'end'],
                            psi0=sv['psi', 'end'],
                            theta0=sv['theta', 'end'])

    for sv_test in [sv, sv_init_start, sv_iinit_end]:
        xo.assert_allclose(sv_test.X, sv.X, atol=5e-14)
        xo.assert_allclose(sv_test.Y, sv.Y, atol=5e-14)
        xo.assert_allclose(sv_test.Z, sv.Z, atol=5e-14)
        xo.assert_allclose(sv_test.phi, sv.phi, atol=5e-14)
        xo.assert_allclose(sv_test.psi, sv.psi, atol=5e-14)
        xo.assert_allclose(sv_test.theta, sv.theta, atol=5e-14)

    tw_back = line.twiss(init=tw, init_at='end')

    xo.assert_allclose(tw_back.x, tw.x, atol=5e-14)
    xo.assert_allclose(tw_back.y, tw.y, atol=5e-14)
    xo.assert_allclose(tw_back.s, tw.s, atol=5e-14)
    xo.assert_allclose(tw_back.zeta, tw.zeta, atol=5e-14)

    line['mb'].rbend_model = 'curved-body'
    tw_curved = line.twiss(betx=1, bety=1)
    sv_curved = line.survey(element0='mid')

    xo.assert_allclose(tw_curved.x, 0, atol=5e-14)
    xo.assert_allclose(tw_curved.y, 0, atol=5e-14)
    xo.assert_allclose(sv_curved['Y', 'mb_entry'], -line['mb'].sagitta, atol=5e-14)
    xo.assert_allclose(sv_curved['Y', 'mid'], 0, atol=5e-14)
    xo.assert_allclose(sv_curved['Y', 'mb_exit'], -line['mb'].sagitta, atol=5e-14)

    sv_curved_init_start = line.survey(element0='start',
                                    X0=sv_curved['X', 'start'],
                                    Y0=sv_curved['Y', 'start'],
                                    Z0=sv_curved['Z', 'start'],
                                    phi0=sv_curved['phi', 'start'],
                                    psi0=sv_curved['psi', 'start'],
                                    theta0=sv_curved['theta', 'start'])
    sv_curved_init_end = line.survey(element0='end',
                                    X0=sv_curved['X', 'end'],
                                    Y0=sv_curved['Y', 'end'],
                                    Z0=sv_curved['Z', 'end'],
                                    phi0=sv_curved['phi', 'end'],
                                    psi0=sv_curved['psi', 'end'],
                                    theta0=sv_curved['theta', 'end'])

    for sv_test in [sv_curved_init_start, sv_curved_init_end]:
        xo.assert_allclose(sv_test.X, sv_curved.X, atol=5e-14)
        xo.assert_allclose(sv_test.Y, sv_curved.Y, atol=5e-14)
        xo.assert_allclose(sv_test.Z, sv_curved.Z, atol=5e-14)
        xo.assert_allclose(sv_test.phi, sv_curved.phi, atol=5e-14)
        xo.assert_allclose(sv_test.psi, sv_curved.psi, atol=5e-14)
        xo.assert_allclose(sv_test.theta, sv_curved.theta, atol=5e-14)

def test_rbend_loading_writing_in_interfaces():

    madx = Madx()

    l_straight = 1.0
    angle = 0.5
    l_arc = l_straight / np.sinc(angle / np.pi / 2) # np.sinc is sin(pi*x)/(pi*x)

    madx.input('''
    b: rbend, l=1.0, angle=0.5;

    ss: sequence, l=5.0, refer=centre;
        b, at=2.5;
    endsequence;
    beam;
    use, sequence=ss;
    twiss, betx=1, bety=1;
    survey;
    ''')
    sv_madx = xt.Table(madx.table.survey, _copy_cols=True)
    xo.assert_allclose(madx.elements['b'].l, l_straight, rtol=1e-12)
    xo.assert_allclose(sv_madx['s', 'b:1'], 2.5 + l_arc / 2, rtol=1e-12) # b si at exit

    # Check cpymad loader
    line = xt.Line.from_madx_sequence(madx.sequence.ss)
    line.remove('ss$start')
    line.remove('ss$end')
    line.particle_ref = xt.Particles(p0c=1e9)
    sv_xs = line.survey()
    xo.assert_allclose(line['b'].length_straight, l_straight, rtol=1e-12)
    xo.assert_allclose(line['b'].length, l_arc, rtol=1e-12)
    xo.assert_allclose(sv_xs['s', 'b'], 2.5 - l_arc / 2, rtol=1e-12) # b si at entrance

    # Check to madx
    mad_str = line.to_madx_sequence(sequence_name='ggg')
    madx2 = Madx()
    madx2.input(mad_str)
    madx2.input('''
    beam;
    use, sequence=ggg;
    twiss, betx=1, bety=1;
    survey;
    ''')
    sv_madx2 = xt.Table(madx2.table.survey, _copy_cols=True)
    xo.assert_allclose(madx2.elements['b'].l, l_straight, rtol=1e-12)

    # Check native madx loader
    env2 = xt.load(string=mad_str, format='madx')
    sv_xs2 = env2.ggg.survey()
    xo.assert_allclose(env2['b'].length_straight, l_straight, rtol=1e-12)
    xo.assert_allclose(env2['b'].length, l_arc, rtol=1e-12)
    xo.assert_allclose(sv_xs2['s', 'b'], 2.5 - l_arc / 2, rtol=1e-12) # b si at entrance

    # Check madng
    sv_ng = line.madng_survey()
    xo.assert_allclose(sv_ng['s', 'b'], 2.5 - l_arc / 2, rtol=1e-12) # b si at entrance

@pytest.mark.parametrize('edge_model', ['linear', 'full'])
def test_rbend_straight_body_chicane_h(edge_model):

    env = xt.Environment()
    env.vars.default_to_zero = True
    line = env.new_line(compose=True)
    line.new('start', 'Marker', at=0.)
    line.new('d1a', 'RBend', length_straight=1.0, k0='k0d1a', anchor='start', at='dz_d1a')
    line.new('d1b', 'RBend', length_straight=1.0, k0='k0d1b', anchor='start', at='dz_d1b')
    line.new('d2',  'RBend', length_straight=1.0, k0='k0d2',  anchor='start', at='dz_d2')
    line.new('end', 'Marker', at='dz_end')

    # ------ measure geometry in the straight reference frame ------

    # Positions in the straight reference frame
    env['dz_d1a'] = 1.
    env['dz_d1b'] = 3.
    env['dz_d2'] = 8.
    env['dz_end'] = 10.

    line.end_compose()
    line.set_particle_ref('proton', p0c=1e9)
    line.configure_drift_model('exact')
    line.set(env.elements.get_table().rows.match(element_type='RBend'),
            model='bend-kick-bend', edge_entry_model=edge_model,
            edge_exit_model=edge_model)

    env['k0d1a'] = 'k0d1'
    env['k0d1b'] = 'k0d1'

    opt = line.match(
        solve=False,
        betx=1, bety=1,
        vary=[xt.VaryList(['k0d1', 'k0d2'], step=1e-5)],
        targets=xt.TargetSet(x=1., px=0.0, at='end'),
    )
    opt.solve()

    # ---- build geometry with curved reference frame ----

    # Twiss in the straight reference system
    tw0 = line.twiss(betx=1, bety=1, strengths=True)

    if edge_model == 'linear':
        # Set fdown angles to match the trajectory (used only for linear edges)
        for nn in ['d1a', 'd1b', 'd2']:
            line[nn].edge_entry_angle_fdown = np.arcsin(tw0['px', nn])
            line[nn].edge_exit_angle_fdown = -np.arcsin(tw0['px', nn + '>>1'])
        tw0 = line.twiss(betx=1, bety=1, strengths=True)
        for nn in ['d1a', 'd1b', 'd2']:
            line[nn].edge_entry_angle_fdown = 0
            line[nn].edge_exit_angle_fdown = 0

    line.regenerate_from_composer()

    # Update positions according to path length
    env['dz_d1a'] = tw0['s', 'd1a'] - tw0['zeta', 'd1a']
    env['dz_d1b'] = tw0['s', 'd1b'] - tw0['zeta', 'd1b']
    env['dz_d2'] = tw0['s', 'd2'] - tw0['zeta', 'd2']
    env['dz_end'] = tw0['s', 'end'] - tw0['zeta', 'end']

    # Introduce magnet curvatures
    for nn in ['d1a', 'd1b', 'd2']:
        line[nn].k0 = 0
        line[nn].k0_from_h = True
        line[nn].rbend_compensate_sagitta = False
        line[nn].rbend_model = 'straight-body'

    d1a_angle_in = np.arcsin(tw0['px', 'd1a'])
    d1b_angle_in = np.arcsin(tw0['px', 'd1b'])
    d2_angle_in  = np.arcsin(tw0['px', 'd2'])
    d1a_angle_out = -d1b_angle_in
    d1b_angle_out = -d2_angle_in
    d2_angle_out  = -np.arcsin(tw0['px', 'end'])

    line['d1a'].angle = d1a_angle_in + d1a_angle_out
    line['d1b'].angle = d1b_angle_in + d1b_angle_out
    line['d2'].angle  = d2_angle_in  + d2_angle_out

    line['d1a'].rbend_angle_diff = d1a_angle_out - d1a_angle_in
    line['d1b'].rbend_angle_diff = d1b_angle_out - d1b_angle_in
    line['d2'].rbend_angle_diff  = d2_angle_out  - d2_angle_in

    # Set rbend shifts
    line['d1a'].rbend_shift += line['d1a']._x0_in - tw0['x', 'd1a']
    line['d1b'].rbend_shift += line['d1b']._x0_in - tw0['x', 'd1b']
    line['d2'].rbend_shift += line['d2']._x0_out - tw0['x', 'end'] # to illustrate that out can be set as well

    line.end_compose()

    sv = line.survey()
    tw = line.twiss(betx=1, bety=1)
    sv_back = line.survey(element0='end', X0=sv.X[-1], Y0=sv.Y[-1], Z0=sv.Z[-1],
                        theta0=sv.theta[-1], phi0=sv.phi[-1], psi0=sv.psi[-1])
    if edge_model == 'linear':
        tw_back = line.twiss(init_at='end', init=tw.get_twiss_init('end'))

    # slice for plot
    l_sliced =line.copy(shallow=True)
    l_sliced.slice_thick_elements(
            slicing_strategies=[
                xt.Strategy(slicing=xt.Uniform(3, mode='thick'))
            ])

    sv_sliced = l_sliced.survey()
    tw_sliced = l_sliced.twiss(betx=1, bety=1)
    sv_sliced_back = l_sliced.survey(element0='end',
                                    X0=sv_sliced.X[-1], Y0=sv_sliced.Y[-1], Z0=sv_sliced.Z[-1],
                                    theta0=sv_sliced.theta[-1], phi0=sv_sliced.phi[-1],
                                    psi0=sv_sliced.psi[-1])
    if edge_model == 'linear':
        tw_sliced_back = l_sliced.twiss(init_at='end',
                                        init=tw_sliced.get_twiss_init('end'))

    # Combine twiss and survey to get actual trajectory
    trajectory = sv_sliced.p0 + tw_sliced.x[:, None] * sv_sliced.ex + tw_sliced.y[:, None] * sv_sliced.ey

    tw0['path_length'] = tw0.s - tw0.zeta
    tw0['diff_path_length'] = np.diff(tw0.path_length, append=tw0.path_length[-1])

    xo.assert_allclose(tw0.path_length, tw.s, atol=1e-14)

    xo.assert_allclose(tw0['diff_path_length', 'd1a'], line['d1a'].length, atol=1e-14)
    xo.assert_allclose(tw0['diff_path_length', 'd1b'], line['d1b'].length, atol=1e-14)
    xo.assert_allclose(tw0['diff_path_length', 'd2'], line['d2'].length, atol=1e-14)

    xo.assert_allclose(tw0['px', 'd1a'], 0, atol=1e-14)
    xo.assert_allclose(tw0['px', 'd1b'], np.sin(line['d1b']._angle_in), atol=1e-14)
    xo.assert_allclose(tw0['px', 'd2'], np.sin(line['d2']._angle_in), atol=1e-14)

    xo.assert_allclose(tw0['px', 'd1b'], -np.sin(line['d1a']._angle_out), atol=1e-14)
    xo.assert_allclose(tw0['px', 'd2'], -np.sin(line['d1b']._angle_out), atol=1e-14)
    xo.assert_allclose(tw0['px', 'end'], -np.sin(line['d2']._angle_out), atol=1e-14)

    xo.assert_allclose(tw0['x', 'd1a'], line['d1a']._x0_in, atol=1e-14)
    xo.assert_allclose(tw0['x', 'd1b'], line['d1b']._x0_in, atol=1e-14)
    xo.assert_allclose(tw0['x', 'd2'], line['d2']._x0_in, atol=1e-14)

    xo.assert_allclose(tw0['x', 'd1a>>1'], line['d1a']._x0_out, atol=1e-14)
    xo.assert_allclose(tw0['x', 'd1b>>1'], line['d1b']._x0_out, atol=1e-14)
    xo.assert_allclose(tw0['x', 'd2>>1'], line['d2']._x0_out, atol=1e-14)

    assert np.all(sv.element_type ==
            ['Marker', 'Drift', 'RBend', 'Drift', 'RBend', 'Drift', 'RBend',
        'Drift', 'Marker', ''])
    xo.assert_allclose(sv.angle, np.array([
            0.        ,  0.        , -0.08249992,  0.        , -0.08306823,
            0.        ,  0.16556815,  0.        ,  0.        ,  0.        ]),
            rtol=1e-7)

    xo.assert_allclose(sv.Z, tw0.s, atol=0, rtol=5e-9)
    xo.assert_allclose(sv.X, tw0.x, atol=0, rtol=3e-8)
    xo.assert_allclose(sv.Y, tw0.y, atol=1e-14)
    xo.assert_allclose(sv.theta, np.arcsin(tw0.px), atol=1e-14)
    xo.assert_allclose(sv.psi, 0., atol=1e-14)
    xo.assert_allclose(sv.phi, 0., atol=1e-14)

    xo.assert_allclose(tw.x, 0, atol=1e-14)
    xo.assert_allclose(tw.zeta, 0, atol=1e-14)
    xo.assert_allclose(tw.y, 0, atol=1e-14)

    xo.assert_allclose(tw.betx[-1], tw0.betx[-1], rtol=1e-10)
    xo.assert_allclose(tw.bety[-1], tw0.bety[-1], rtol=1e-10)
    xo.assert_allclose(tw.alfx[-1], tw0.alfx[-1], rtol=1e-10)
    xo.assert_allclose(tw.alfy[-1], tw0.alfy[-1], rtol=1e-10)
    xo.assert_allclose(tw.dx[-1], tw0.dx[-1], rtol=1e-10)
    xo.assert_allclose(tw.dpx[-1], tw0.dpx[-1], atol=1e-10)

    if edge_model == 'linear':
        xo.assert_allclose(tw_back.s, tw.s, atol=1e-14)
        xo.assert_allclose(tw_back.x, tw.x, atol=1e-14)
        xo.assert_allclose(tw_back.y, tw.y, atol=1e-14)
        xo.assert_allclose(tw_back.betx, tw.betx, rtol=3e-10)
        xo.assert_allclose(tw_back.bety, tw.bety, rtol=3e-10)
        xo.assert_allclose(tw_back.alfx, tw.alfx, atol=1e-8)
        xo.assert_allclose(tw_back.alfy, tw.alfy, atol=1e-8)
        xo.assert_allclose(tw_back.dx, tw.dx, atol=1e-9)
        xo.assert_allclose(tw_back.dpx, tw.dpx, atol=1e-9)

    xo.assert_allclose(sv_back.s, sv.s, atol=1e-14)
    xo.assert_allclose(sv_back.X, sv.X, atol=1e-14)
    xo.assert_allclose(sv_back.Y, sv.Y, atol=1e-14)
    xo.assert_allclose(sv_back.Z, sv.Z, atol=1e-14)
    xo.assert_allclose(sv_back.theta, sv.theta, atol=1e-14)
    xo.assert_allclose(sv_back.phi, sv.phi, atol=1e-14)
    xo.assert_allclose(sv_back.psi, sv.psi, atol=1e-14)
    xo.assert_allclose(sv.angle, sv.angle, atol=1e-14)

    sv_sliced.cols['s angle theta X'].show()
    # name                       s         angle         theta             X
    # start                      0             0             0             0
    # ||drift_1                  0             0             0             0
    # d1a_entry                  1             0             0             0
    # d1a..entry_map             1             0             0             0
    # d1a..0                     1             0             0             0
    # d1a..1               1.33371             0             0             0
    # d1a..2               1.66742             0             0             0
    # d1a..exit_map        2.00114    -0.0824999             0             0
    # d1a_exit             2.00114             0     0.0824999     0.0412734
    # ||drift_3            2.00114             0     0.0824999     0.0412734
    # d1b_entry            3.00455             0     0.0824999      0.123961
    # d1b..entry_map       3.00455     0.0824999     0.0824999      0.123961
    # d1b..0               3.00455             0  -1.14732e-17  -1.38778e-17
    # d1b..1               3.34056             0  -1.14732e-17  -1.77022e-17
    # d1b..2               3.67657             0  -1.14732e-17  -2.15266e-17
    # d1b..exit_map        4.01258     -0.165568  -1.14732e-17   -2.5351e-17
    # d1b_exit             4.01258             0      0.165568      0.248635
    # ||drift_4            4.01258             0      0.165568      0.248635
    # d2_entry             8.06804             0      0.165568      0.917026
    # d2..entry_map        8.06804      0.165568      0.165568      0.917026
    # d2..0                8.06804             0  -9.64632e-18   1.11022e-16
    # d2..1                 8.4029             0  -9.64632e-18   1.07807e-16
    # d2..2                8.73776             0  -9.64632e-18   1.04591e-16
    # d2..exit_map         9.07262   1.38778e-17  -9.64632e-18   1.01376e-16
    # d2_exit              9.07262             0  -9.64632e-18             1
    # ||drift_5            9.07262             0  -9.64632e-18             1
    # end                  10.0726             0  -9.64632e-18             1
    # _end_point           10.0726             0  -9.64632e-18             1

    xo.assert_allclose(sv_sliced.s[-1], tw0.path_length[-1], atol=0, rtol=1e-14)
    xo.assert_allclose(sv_sliced.X[-1], tw0.x[-1], atol=0, rtol=1e-14)
    xo.assert_allclose(sv_sliced.Y, 0, atol=1e-14)

    assert np.all(sv_sliced.element_type ==
        np.array(['Marker', 'Drift', 'Marker', 'ThinSliceRBendEntry',
        'ThickSliceRBend', 'ThickSliceRBend', 'ThickSliceRBend',
        'ThinSliceRBendExit', 'Marker', 'Drift', 'Marker',
        'ThinSliceRBendEntry', 'ThickSliceRBend', 'ThickSliceRBend',
        'ThickSliceRBend', 'ThinSliceRBendExit', 'Marker', 'Drift',
        'Marker', 'ThinSliceRBendEntry', 'ThickSliceRBend',
        'ThickSliceRBend', 'ThickSliceRBend', 'ThinSliceRBendExit',
        'Marker', 'Drift', 'Marker', '']))

    xo.assert_allclose(sv_sliced.angle, np.array([
            0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
            0.00000000e+00,  0.00000000e+00,  0.00000000e+00, -8.24999219e-02,
            0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  8.24999219e-02,
            0.00000000e+00,  0.00000000e+00,  0.00000000e+00, -1.65568148e-01,
            0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.65568148e-01,
            0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
            0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00]),
            rtol=1e-8, atol=1e-14)

    xo.assert_allclose(sv_sliced['s', 'd1a..entry_map'], sv['s', 'd1a'], atol=1e-14)
    xo.assert_allclose(sv_sliced['s', 'd1b..entry_map'], sv['s', 'd1b'], atol=1e-14)
    xo.assert_allclose(sv_sliced['s', 'd2..entry_map'],  sv['s', 'd2'],  atol=1e-14)
    xo.assert_allclose(sv_sliced['s', 'd1a..exit_map>>1'], sv['s', 'd1a>>1'], atol=1e-14)
    xo.assert_allclose(sv_sliced['s', 'd1b..exit_map>>1'], sv['s', 'd1b>>1'], atol=1e-14)
    xo.assert_allclose(sv_sliced['s', 'd2..exit_map>>1'],  sv['s', 'd2>>1'],  atol=1e-14)

    xo.assert_allclose(sv_sliced['theta', 'd1a..entry_map'], sv['theta', 'd1a'], atol=1e-14)
    xo.assert_allclose(sv_sliced['theta', 'd1b..entry_map'], sv['theta', 'd1b'], atol=1e-14)
    xo.assert_allclose(sv_sliced['theta', 'd2..entry_map'],  sv['theta', 'd2'],  atol=1e-14)
    xo.assert_allclose(sv_sliced['theta', 'd1a..exit_map>>1'], sv['theta', 'd1a>>1'], atol=1e-14)
    xo.assert_allclose(sv_sliced['theta', 'd1b..exit_map>>1'], sv['theta', 'd1b>>1'], atol=1e-14)
    xo.assert_allclose(sv_sliced['theta', 'd2..exit_map>>1'],  sv['theta', 'd2>>1'],  atol=1e-14)

    xo.assert_allclose(sv_sliced['X', 'd1a..entry_map'], sv['X', 'd1a'], atol=1e-14)
    xo.assert_allclose(sv_sliced['X', 'd1b..entry_map'], sv['X', 'd1b'], atol=1e-14)
    xo.assert_allclose(sv_sliced['X', 'd2..entry_map'],  sv['X', 'd2'],  atol=1e-14)
    xo.assert_allclose(sv_sliced['X', 'd1a..exit_map>>1'], sv['X', 'd1a>>1'], atol=1e-14)
    xo.assert_allclose(sv_sliced['X', 'd1b..exit_map>>1'], sv['X', 'd1b>>1'], atol=1e-14)
    xo.assert_allclose(sv_sliced['X', 'd2..exit_map>>1'],  sv['X', 'd2>>1'],  atol=1e-14)

    xo.assert_allclose(sv_sliced['Z', 'd1a..entry_map'], sv['Z', 'd1a'], atol=1e-14)
    xo.assert_allclose(sv_sliced['Z', 'd1b..entry_map'], sv['Z', 'd1b'], atol=1e-14)
    xo.assert_allclose(sv_sliced['Z', 'd2..entry_map'],  sv['Z', 'd2'],  atol=1e-14)
    xo.assert_allclose(sv_sliced['Z', 'd1a..exit_map>>1'], sv['Z', 'd1a>>1'], atol=1e-14)
    xo.assert_allclose(sv_sliced['Z', 'd1b..exit_map>>1'], sv['Z', 'd1b>>1'], atol=1e-14)
    xo.assert_allclose(sv_sliced['Z', 'd2..exit_map>>1'],  sv['Z', 'd2>>1'],  atol=1e-14)

    xo.assert_allclose(sv_sliced['theta', 'd1a..entry_map>>1'], 0., atol=1e-14)
    xo.assert_allclose(sv_sliced['theta', 'd1b..entry_map>>1'], 0., atol=1e-14)
    xo.assert_allclose(sv_sliced['theta', 'd2..entry_map>>1'],  0.,  atol=1e-14)
    xo.assert_allclose(sv_sliced['theta', 'd1a..exit_map'], 0., atol=1e-14)
    xo.assert_allclose(sv_sliced['theta', 'd1b..exit_map'], 0., atol=1e-14)
    xo.assert_allclose(sv_sliced['theta', 'd2..exit_map'],  0.,  atol=1e-14)

    xo.assert_allclose(sv_sliced['X', 'd1a..entry_map>>1'], 0., atol=1e-14)
    xo.assert_allclose(sv_sliced['X', 'd1b..entry_map>>1'], 0., atol=1e-14)
    xo.assert_allclose(sv_sliced['X', 'd2..entry_map>>1'],  0.,  atol=1e-14)
    xo.assert_allclose(sv_sliced['X', 'd1a..exit_map'], 0., atol=1e-14)
    xo.assert_allclose(sv_sliced['X', 'd1b..exit_map'], 0., atol=1e-14)
    xo.assert_allclose(sv_sliced['X', 'd2..exit_map'],  0.,  atol=1e-14)

    xo.assert_allclose(sv_sliced_back.s, sv_sliced.s, atol=1e-14)
    xo.assert_allclose(sv_sliced_back.X, sv_sliced.X, atol=1e-14)
    xo.assert_allclose(sv_sliced_back.Y, sv_sliced.Y, atol=1e-14)
    xo.assert_allclose(sv_sliced_back.Z, sv_sliced.Z, atol=1e-14)
    xo.assert_allclose(sv_sliced_back.theta, sv_sliced.theta, atol=1e-14)
    xo.assert_allclose(sv_sliced_back.phi, sv_sliced.phi, atol=1e-14)
    xo.assert_allclose(sv_sliced_back.psi, sv_sliced.psi, atol=1e-14)
    xo.assert_allclose(sv_sliced_back.angle, sv_sliced.angle, atol=1e-14)

    # Twiss checks

    xo.assert_allclose(tw_sliced.s, sv_sliced.s, atol=0, rtol=1e-14)

    xo.assert_allclose(tw_sliced['x', 'd1a..entry_map'], 0, atol=1e-14)
    xo.assert_allclose(tw_sliced['x', 'd1b..entry_map'], 0, atol=1e-14)
    xo.assert_allclose(tw_sliced['x', 'd2..entry_map'],  0,  atol=1e-14)
    xo.assert_allclose(tw_sliced['x', 'd1a..exit_map>>1'], 0, atol=1e-14)
    xo.assert_allclose(tw_sliced['x', 'd1b..exit_map>>1'], 0, atol=1e-14)
    xo.assert_allclose(tw_sliced['x', 'd2..exit_map>>1'],  0,  atol=1e-14)

    xo.assert_allclose(tw_sliced['px', 'd1a..entry_map'], 0, atol=1e-14)
    xo.assert_allclose(tw_sliced['px', 'd1b..entry_map'], 0, atol=1e-14)
    xo.assert_allclose(tw_sliced['px', 'd2..entry_map'],  0,  atol=1e-14)
    xo.assert_allclose(tw_sliced['px', 'd1a..exit_map>>1'], 0, atol=1e-14)
    xo.assert_allclose(tw_sliced['px', 'd1b..exit_map>>1'], 0, atol=1e-14)
    xo.assert_allclose(tw_sliced['px', 'd2..exit_map>>1'],  0,  atol=1e-14)

    xo.assert_allclose(tw_sliced['x', 'd1a..entry_map>>1'], tw0['x','d1a'], atol=1e-14)
    xo.assert_allclose(tw_sliced['x', 'd1b..entry_map>>1'], tw0['x','d1b'], atol=1e-14)
    xo.assert_allclose(tw_sliced['x', 'd2..entry_map>>1'],  tw0['x','d2'],  atol=1e-14)
    xo.assert_allclose(tw_sliced['x', 'd1a..exit_map'], tw0['x','d1a>>1'], atol=1e-14)
    xo.assert_allclose(tw_sliced['x', 'd1b..exit_map'], tw0['x','d1b>>1'], atol=1e-14)
    xo.assert_allclose(tw_sliced['x', 'd2..exit_map'],  tw0['x','d2>>1'],  atol=1e-14)

    xo.assert_allclose(tw_sliced.betx[-1], tw0.betx[-1], rtol=1e-10)
    xo.assert_allclose(tw_sliced.bety[-1], tw0.bety[-1], rtol=1e-10)
    xo.assert_allclose(tw_sliced.alfx[-1], tw0.alfx[-1], rtol=1e-10)
    xo.assert_allclose(tw_sliced.alfy[-1], tw0.alfy[-1], rtol=1e-10)
    xo.assert_allclose(tw_sliced.dx[-1], tw0.dx[-1], rtol=1e-10)
    xo.assert_allclose(tw_sliced.dpx[-1], tw0.dpx[-1], atol=1e-10)

    if edge_model == 'linear':
        xo.assert_allclose(tw_sliced_back.s, tw_sliced.s, atol=1e-14)
        xo.assert_allclose(tw_sliced_back.x, tw_sliced.x, atol=1e-14)
        xo.assert_allclose(tw_sliced_back.y, tw_sliced.y, atol=1e-14)
        xo.assert_allclose(tw_sliced_back.betx, tw_sliced.betx, rtol=1e-8)
        xo.assert_allclose(tw_sliced_back.bety, tw_sliced.bety, rtol=1e-8)
        xo.assert_allclose(tw_sliced_back.alfx, tw_sliced.alfx, atol=1e-8)
        xo.assert_allclose(tw_sliced_back.alfy, tw_sliced.alfy, atol=1e-8)
        xo.assert_allclose(tw_sliced_back.dx, tw_sliced.dx, atol=1e-9)
        xo.assert_allclose(tw_sliced_back.dpx, tw_sliced.dpx, atol=1e-9)

@pytest.mark.parametrize('edge_model', ['linear', 'full'])
def test_rbend_straight_body_chicane_v(edge_model):

    env = xt.Environment()
    env.vars.default_to_zero = True
    line = env.new_line(compose=True)
    line.new('start', 'Marker', at=0.)
    line.new('d1a', 'RBend', rot_s_rad=np.pi/2, length_straight=1.0, k0='k0d1a', anchor='start', at='dz_d1a')
    line.new('d1b', 'RBend', rot_s_rad=np.pi/2, length_straight=1.0, k0='k0d1b', anchor='start', at='dz_d1b')
    line.new('d2',  'RBend', rot_s_rad=np.pi/2, length_straight=1.0, k0='k0d2',  anchor='start', at='dz_d2')
    line.new('end', 'Marker', at='dz_end')

    # ------ measure geometry in the straight reference frame ------

    # Positions in the straight reference frame
    env['dz_d1a'] = 1.
    env['dz_d1b'] = 3.
    env['dz_d2'] = 8.
    env['dz_end'] = 10.

    line.end_compose()
    line.set_particle_ref('proton', p0c=1e9)
    line.configure_drift_model('exact')
    line.set(env.elements.get_table().rows.match(element_type='RBend'),
            model='bend-kick-bend', edge_entry_model=edge_model,
            edge_exit_model=edge_model)

    env['k0d1a'] = 'k0d1'
    env['k0d1b'] = 'k0d1'

    opt = line.match(
        solve=False,
        betx=1, bety=1,
        vary=[xt.VaryList(['k0d1', 'k0d2'], step=1e-5)],
        targets=xt.TargetSet(y=1., py=0.0, at='end'),
    )
    opt.solve()

    # ---- build geometry with curved reference frame ----

    # Twiss in the straight reference system
    tw0 = line.twiss(betx=1, bety=1, strengths=True)

    if edge_model == 'linear':
        # Set fdown angles to match the trajectory (used only for linear edges)
        for nn in ['d1a', 'd1b', 'd2']:
            line[nn].edge_entry_angle_fdown = np.arcsin(tw0['py', nn])
            line[nn].edge_exit_angle_fdown = -np.arcsin(tw0['py', nn + '>>1'])
        tw0 = line.twiss(betx=1, bety=1, strengths=True)
        for nn in ['d1a', 'd1b', 'd2']:
            line[nn].edge_entry_angle_fdown = 0
            line[nn].edge_exit_angle_fdown = 0

    line.regenerate_from_composer()

    # Update positions according to path length
    env['dz_d1a'] = tw0['s', 'd1a'] - tw0['zeta', 'd1a']
    env['dz_d1b'] = tw0['s', 'd1b'] - tw0['zeta', 'd1b']
    env['dz_d2'] = tw0['s', 'd2'] - tw0['zeta', 'd2']
    env['dz_end'] = tw0['s', 'end'] - tw0['zeta', 'end']

    # Introduce magnet curvatures
    for nn in ['d1a', 'd1b', 'd2']:
        line[nn].k0 = 0
        line[nn].k0_from_h = True
        line[nn].rbend_compensate_sagitta = False
        line[nn].rbend_model = 'straight-body'

    d1a_angle_in = np.arcsin(tw0['py', 'd1a'])
    d1b_angle_in = np.arcsin(tw0['py', 'd1b'])
    d2_angle_in  = np.arcsin(tw0['py', 'd2'])
    d1a_angle_out = -d1b_angle_in
    d1b_angle_out = -d2_angle_in
    d2_angle_out  = -np.arcsin(tw0['py', 'end'])

    line['d1a'].angle = d1a_angle_in + d1a_angle_out
    line['d1b'].angle = d1b_angle_in + d1b_angle_out
    line['d2'].angle  = d2_angle_in  + d2_angle_out

    line['d1a'].rbend_angle_diff = d1a_angle_out - d1a_angle_in
    line['d1b'].rbend_angle_diff = d1b_angle_out - d1b_angle_in
    line['d2'].rbend_angle_diff  = d2_angle_out  - d2_angle_in

    # Set rbend shifts
    line['d1a'].rbend_shift += line['d1a']._x0_in - tw0['y', 'd1a']
    line['d1b'].rbend_shift += line['d1b']._x0_in - tw0['y', 'd1b']
    line['d2'].rbend_shift += line['d2']._x0_out - tw0['y', 'end'] # to illustrate that out can be set as well

    line.end_compose()

    sv = line.survey()
    tw = line.twiss(betx=1, bety=1)
    sv_back = line.survey(element0='end', X0=sv.X[-1], Y0=sv.Y[-1], Z0=sv.Z[-1],
                        theta0=sv.theta[-1], phi0=sv.phi[-1], psi0=sv.psi[-1])
    if edge_model == 'linear':
        tw_back = line.twiss(init_at='end', init=tw.get_twiss_init('end'))

    # slice for plot
    l_sliced =line.copy(shallow=True)
    l_sliced.slice_thick_elements(
            slicing_strategies=[
                xt.Strategy(slicing=xt.Uniform(3, mode='thick'))
            ])

    sv_sliced = l_sliced.survey()
    tw_sliced = l_sliced.twiss(betx=1, bety=1)
    sv_sliced_back = l_sliced.survey(element0='end',
                                    X0=sv_sliced.X[-1], Y0=sv_sliced.Y[-1], Z0=sv_sliced.Z[-1],
                                    theta0=sv_sliced.theta[-1], phi0=sv_sliced.phi[-1],
                                    psi0=sv_sliced.psi[-1])
    if edge_model == 'linear':
        tw_sliced_back = l_sliced.twiss(init_at='end',
                                        init=tw_sliced.get_twiss_init('end'))

    # Combine twiss and survey to get actual trajectory
    trajectory = sv_sliced.p0 + tw_sliced.x[:, None] * sv_sliced.ex + tw_sliced.y[:, None] * sv_sliced.ey

    tw0['path_length'] = tw0.s - tw0.zeta
    tw0['diff_path_length'] = np.diff(tw0.path_length, append=tw0.path_length[-1])

    xo.assert_allclose(tw0.path_length, tw.s, atol=1e-14)

    xo.assert_allclose(tw0['diff_path_length', 'd1a'], line['d1a'].length, atol=1e-14)
    xo.assert_allclose(tw0['diff_path_length', 'd1b'], line['d1b'].length, atol=1e-14)
    xo.assert_allclose(tw0['diff_path_length', 'd2'], line['d2'].length, atol=1e-14)

    xo.assert_allclose(tw0['py', 'd1a'], 0, atol=1e-14)
    xo.assert_allclose(tw0['py', 'd1b'], np.sin(line['d1b']._angle_in), atol=1e-14)
    xo.assert_allclose(tw0['py', 'd2'], np.sin(line['d2']._angle_in), atol=1e-14)

    xo.assert_allclose(tw0['py', 'd1b'], -np.sin(line['d1a']._angle_out), atol=1e-14)
    xo.assert_allclose(tw0['py', 'd2'], -np.sin(line['d1b']._angle_out), atol=1e-14)
    xo.assert_allclose(tw0['py', 'end'], -np.sin(line['d2']._angle_out), atol=1e-14)

    xo.assert_allclose(tw0['y', 'd1a'], line['d1a']._x0_in, atol=1e-14)
    xo.assert_allclose(tw0['y', 'd1b'], line['d1b']._x0_in, atol=1e-14)
    xo.assert_allclose(tw0['y', 'd2'], line['d2']._x0_in, atol=1e-14)

    xo.assert_allclose(tw0['y', 'd1a>>1'], line['d1a']._x0_out, atol=1e-14)
    xo.assert_allclose(tw0['y', 'd1b>>1'], line['d1b']._x0_out, atol=1e-14)
    xo.assert_allclose(tw0['y', 'd2>>1'], line['d2']._x0_out, atol=1e-14)

    assert np.all(sv.element_type ==
            ['Marker', 'Drift', 'RBend', 'Drift', 'RBend', 'Drift', 'RBend',
        'Drift', 'Marker', ''])
    xo.assert_allclose(sv.angle, np.array([
            0.        ,  0.        , -0.08249992,  0.        , -0.08306823,
            0.        ,  0.16556815,  0.        ,  0.        ,  0.        ]),
            rtol=1e-7)

    xo.assert_allclose(sv.Z, tw0.s, atol=0, rtol=5e-9)
    xo.assert_allclose(sv.X, tw0.x, atol=0, rtol=3e-8)
    xo.assert_allclose(sv.Y, tw0.y, atol=1e-14)
    xo.assert_allclose(sv.theta, 0., atol=1e-14)
    xo.assert_allclose(sv.psi, 0., atol=1e-14)
    xo.assert_allclose(sv.phi, np.arcsin(tw0.py), atol=1e-14)

    xo.assert_allclose(tw.x, 0, atol=1e-14)
    xo.assert_allclose(tw.zeta, 0, atol=1e-14)
    xo.assert_allclose(tw.y, 0, atol=1e-14)

    xo.assert_allclose(tw.betx[-1], tw0.betx[-1], rtol=1e-10)
    xo.assert_allclose(tw.bety[-1], tw0.bety[-1], rtol=1e-10)
    xo.assert_allclose(tw.alfx[-1], tw0.alfx[-1], rtol=1e-10)
    xo.assert_allclose(tw.alfy[-1], tw0.alfy[-1], rtol=1e-10)
    xo.assert_allclose(tw.dx[-1], tw0.dx[-1], rtol=1e-10)
    xo.assert_allclose(tw.dpx[-1], tw0.dpx[-1], atol=1e-10)

    if edge_model == 'linear':
        xo.assert_allclose(tw_back.s, tw.s, atol=1e-14)
        xo.assert_allclose(tw_back.x, tw.x, atol=1e-14)
        xo.assert_allclose(tw_back.y, tw.y, atol=1e-14)
        xo.assert_allclose(tw_back.betx, tw.betx, rtol=5e-10)
        xo.assert_allclose(tw_back.bety, tw.bety, rtol=5e-10)
        xo.assert_allclose(tw_back.alfx, tw.alfx, atol=1e-8)
        xo.assert_allclose(tw_back.alfy, tw.alfy, atol=1e-8)
        xo.assert_allclose(tw_back.dx, tw.dx, atol=1e-9)
        xo.assert_allclose(tw_back.dpx, tw.dpx, atol=1e-9)

    xo.assert_allclose(sv_back.s, sv.s, atol=1e-14)
    xo.assert_allclose(sv_back.X, sv.X, atol=1e-14)
    xo.assert_allclose(sv_back.Y, sv.Y, atol=1e-14)
    xo.assert_allclose(sv_back.Z, sv.Z, atol=1e-14)
    xo.assert_allclose(sv_back.theta, sv.theta, atol=1e-14)
    xo.assert_allclose(sv_back.phi, sv.phi, atol=1e-14)
    xo.assert_allclose(sv_back.psi, sv.psi, atol=1e-14)
    xo.assert_allclose(sv.angle, sv.angle, atol=1e-14)

    sv_sliced.cols['s angle theta X'].show()

    xo.assert_allclose(sv_sliced.s[-1], tw0.path_length[-1], atol=0, rtol=1e-14)
    xo.assert_allclose(sv_sliced.Y[-1], tw0.y[-1], atol=0, rtol=1e-14)
    xo.assert_allclose(sv_sliced.X, 0, atol=1e-14)

    assert np.all(sv_sliced.element_type ==
        np.array(['Marker', 'Drift', 'Marker', 'ThinSliceRBendEntry',
        'ThickSliceRBend', 'ThickSliceRBend', 'ThickSliceRBend',
        'ThinSliceRBendExit', 'Marker', 'Drift', 'Marker',
        'ThinSliceRBendEntry', 'ThickSliceRBend', 'ThickSliceRBend',
        'ThickSliceRBend', 'ThinSliceRBendExit', 'Marker', 'Drift',
        'Marker', 'ThinSliceRBendEntry', 'ThickSliceRBend',
        'ThickSliceRBend', 'ThickSliceRBend', 'ThinSliceRBendExit',
        'Marker', 'Drift', 'Marker', '']))

    xo.assert_allclose(sv_sliced.angle, np.array([
            0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
            0.00000000e+00,  0.00000000e+00,  0.00000000e+00, -8.24999219e-02,
            0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  8.24999219e-02,
            0.00000000e+00,  0.00000000e+00,  0.00000000e+00, -1.65568148e-01,
            0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.65568148e-01,
            0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
            0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00]),
            rtol=1e-8, atol=1e-14)

    xo.assert_allclose(sv_sliced['s', 'd1a..entry_map'], sv['s', 'd1a'], atol=1e-14)
    xo.assert_allclose(sv_sliced['s', 'd1b..entry_map'], sv['s', 'd1b'], atol=1e-14)
    xo.assert_allclose(sv_sliced['s', 'd2..entry_map'],  sv['s', 'd2'],  atol=1e-14)
    xo.assert_allclose(sv_sliced['s', 'd1a..exit_map>>1'], sv['s', 'd1a>>1'], atol=1e-14)
    xo.assert_allclose(sv_sliced['s', 'd1b..exit_map>>1'], sv['s', 'd1b>>1'], atol=1e-14)
    xo.assert_allclose(sv_sliced['s', 'd2..exit_map>>1'],  sv['s', 'd2>>1'],  atol=1e-14)

    xo.assert_allclose(sv_sliced['theta', 'd1a..entry_map'], sv['theta', 'd1a'], atol=1e-14)
    xo.assert_allclose(sv_sliced['theta', 'd1b..entry_map'], sv['theta', 'd1b'], atol=1e-14)
    xo.assert_allclose(sv_sliced['theta', 'd2..entry_map'],  sv['theta', 'd2'],  atol=1e-14)
    xo.assert_allclose(sv_sliced['theta', 'd1a..exit_map>>1'], sv['theta', 'd1a>>1'], atol=1e-14)
    xo.assert_allclose(sv_sliced['theta', 'd1b..exit_map>>1'], sv['theta', 'd1b>>1'], atol=1e-14)
    xo.assert_allclose(sv_sliced['theta', 'd2..exit_map>>1'],  sv['theta', 'd2>>1'],  atol=1e-14)

    xo.assert_allclose(sv_sliced['X', 'd1a..entry_map'], sv['X', 'd1a'], atol=1e-14)
    xo.assert_allclose(sv_sliced['X', 'd1b..entry_map'], sv['X', 'd1b'], atol=1e-14)
    xo.assert_allclose(sv_sliced['X', 'd2..entry_map'],  sv['X', 'd2'],  atol=1e-14)
    xo.assert_allclose(sv_sliced['X', 'd1a..exit_map>>1'], sv['X', 'd1a>>1'], atol=1e-14)
    xo.assert_allclose(sv_sliced['X', 'd1b..exit_map>>1'], sv['X', 'd1b>>1'], atol=1e-14)
    xo.assert_allclose(sv_sliced['X', 'd2..exit_map>>1'],  sv['X', 'd2>>1'],  atol=1e-14)

    xo.assert_allclose(sv_sliced['Z', 'd1a..entry_map'], sv['Z', 'd1a'], atol=1e-14)
    xo.assert_allclose(sv_sliced['Z', 'd1b..entry_map'], sv['Z', 'd1b'], atol=1e-14)
    xo.assert_allclose(sv_sliced['Z', 'd2..entry_map'],  sv['Z', 'd2'],  atol=1e-14)
    xo.assert_allclose(sv_sliced['Z', 'd1a..exit_map>>1'], sv['Z', 'd1a>>1'], atol=1e-14)
    xo.assert_allclose(sv_sliced['Z', 'd1b..exit_map>>1'], sv['Z', 'd1b>>1'], atol=1e-14)
    xo.assert_allclose(sv_sliced['Z', 'd2..exit_map>>1'],  sv['Z', 'd2>>1'],  atol=1e-14)

    xo.assert_allclose(sv_sliced['theta', 'd1a..entry_map>>1'], 0., atol=1e-14)
    xo.assert_allclose(sv_sliced['theta', 'd1b..entry_map>>1'], 0., atol=1e-14)
    xo.assert_allclose(sv_sliced['theta', 'd2..entry_map>>1'],  0.,  atol=1e-14)
    xo.assert_allclose(sv_sliced['theta', 'd1a..exit_map'], 0., atol=1e-14)
    xo.assert_allclose(sv_sliced['theta', 'd1b..exit_map'], 0., atol=1e-14)
    xo.assert_allclose(sv_sliced['theta', 'd2..exit_map'],  0.,  atol=1e-14)

    xo.assert_allclose(sv_sliced['X', 'd1a..entry_map>>1'], 0., atol=1e-14)
    xo.assert_allclose(sv_sliced['X', 'd1b..entry_map>>1'], 0., atol=1e-14)
    xo.assert_allclose(sv_sliced['X', 'd2..entry_map>>1'],  0.,  atol=1e-14)
    xo.assert_allclose(sv_sliced['X', 'd1a..exit_map'], 0., atol=1e-14)
    xo.assert_allclose(sv_sliced['X', 'd1b..exit_map'], 0., atol=1e-14)
    xo.assert_allclose(sv_sliced['X', 'd2..exit_map'],  0.,  atol=1e-14)

    xo.assert_allclose(sv_sliced_back.s, sv_sliced.s, atol=1e-14)
    xo.assert_allclose(sv_sliced_back.X, sv_sliced.X, atol=1e-14)
    xo.assert_allclose(sv_sliced_back.Y, sv_sliced.Y, atol=1e-14)
    xo.assert_allclose(sv_sliced_back.Z, sv_sliced.Z, atol=1e-14)
    xo.assert_allclose(sv_sliced_back.theta, sv_sliced.theta, atol=1e-14)
    xo.assert_allclose(sv_sliced_back.phi, sv_sliced.phi, atol=1e-14)
    xo.assert_allclose(sv_sliced_back.psi, sv_sliced.psi, atol=1e-14)
    xo.assert_allclose(sv_sliced_back.angle, sv_sliced.angle, atol=1e-14)

    # Twiss checks

    xo.assert_allclose(tw_sliced.s, sv_sliced.s, atol=0, rtol=1e-14)

    xo.assert_allclose(tw_sliced['x', 'd1a..entry_map'], 0, atol=1e-14)
    xo.assert_allclose(tw_sliced['x', 'd1b..entry_map'], 0, atol=1e-14)
    xo.assert_allclose(tw_sliced['x', 'd2..entry_map'],  0,  atol=1e-14)
    xo.assert_allclose(tw_sliced['x', 'd1a..exit_map>>1'], 0, atol=1e-14)
    xo.assert_allclose(tw_sliced['x', 'd1b..exit_map>>1'], 0, atol=1e-14)
    xo.assert_allclose(tw_sliced['x', 'd2..exit_map>>1'],  0,  atol=1e-14)

    xo.assert_allclose(tw_sliced['px', 'd1a..entry_map'], 0, atol=1e-14)
    xo.assert_allclose(tw_sliced['px', 'd1b..entry_map'], 0, atol=1e-14)
    xo.assert_allclose(tw_sliced['px', 'd2..entry_map'],  0,  atol=1e-14)
    xo.assert_allclose(tw_sliced['px', 'd1a..exit_map>>1'], 0, atol=1e-14)
    xo.assert_allclose(tw_sliced['px', 'd1b..exit_map>>1'], 0, atol=1e-14)
    xo.assert_allclose(tw_sliced['px', 'd2..exit_map>>1'],  0,  atol=1e-14)

    xo.assert_allclose(tw_sliced['x', 'd1a..entry_map>>1'], tw0['x','d1a'], atol=1e-14)
    xo.assert_allclose(tw_sliced['x', 'd1b..entry_map>>1'], tw0['x','d1b'], atol=1e-14)
    xo.assert_allclose(tw_sliced['x', 'd2..entry_map>>1'],  tw0['x','d2'],  atol=1e-14)
    xo.assert_allclose(tw_sliced['x', 'd1a..exit_map'], tw0['x','d1a>>1'], atol=1e-14)
    xo.assert_allclose(tw_sliced['x', 'd1b..exit_map'], tw0['x','d1b>>1'], atol=1e-14)
    xo.assert_allclose(tw_sliced['x', 'd2..exit_map'],  tw0['x','d2>>1'],  atol=1e-14)

    xo.assert_allclose(tw_sliced.betx[-1], tw0.betx[-1], rtol=1e-10)
    xo.assert_allclose(tw_sliced.bety[-1], tw0.bety[-1], rtol=1e-10)
    xo.assert_allclose(tw_sliced.alfx[-1], tw0.alfx[-1], rtol=1e-10)
    xo.assert_allclose(tw_sliced.alfy[-1], tw0.alfy[-1], rtol=1e-10)
    xo.assert_allclose(tw_sliced.dx[-1], tw0.dx[-1], rtol=1e-10)
    xo.assert_allclose(tw_sliced.dpx[-1], tw0.dpx[-1], atol=1e-10)

    if edge_model == 'linear':
        xo.assert_allclose(tw_sliced_back.s, tw_sliced.s, atol=1e-14)
        xo.assert_allclose(tw_sliced_back.x, tw_sliced.x, atol=1e-14)
        xo.assert_allclose(tw_sliced_back.y, tw_sliced.y, atol=1e-14)
        xo.assert_allclose(tw_sliced_back.betx, tw_sliced.betx, rtol=1e-8)
        xo.assert_allclose(tw_sliced_back.bety, tw_sliced.bety, rtol=1e-8)
        xo.assert_allclose(tw_sliced_back.alfx, tw_sliced.alfx, atol=1e-8)
        xo.assert_allclose(tw_sliced_back.alfy, tw_sliced.alfy, atol=1e-8)
        xo.assert_allclose(tw_sliced_back.dx, tw_sliced.dx, atol=1e-9)
        xo.assert_allclose(tw_sliced_back.dpx, tw_sliced.dpx, atol=1e-9)
