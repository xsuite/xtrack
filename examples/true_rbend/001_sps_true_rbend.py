import xtrack as xt
import xobjects as xo

env = xt.load('../../test_data/sps_thick/sps.seq')
env.vars.load('../../test_data/sps_thick/lhc_q20.str')

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
