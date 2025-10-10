import xtrack as xt
import xobjects as xo

env = xt.Environment()
env.particle_ref = xt.Particles(p0c=7e12)
line = env.new_line(length=5, components=[
    env.new('b', xt.Bend, k0=0.1, h=0.2, length=1.0, at=1.5),
    env.new('end', xt.Marker, at=5.0)
])

line['b'].rot_shift_anchor=0.5
line['b'].rot_x_rad=0.1
line['b'].rot_y_rad=0.2
line['b'].rot_s_rad=0.3
line['b'].rot_s_rad_no_frame=0.1
line['b'].shift_x=0.001
line['b'].shift_y=-0.002
line['b'].shift_s=0.003

tw = line.twiss(betx=1, bety=2, delta=1e-2)
print("tw:", tw)
tw_back = line.twiss(init_at='end', init=tw)
print("tw_back:", tw_back)

xo.assert_allclose(tw_back.x, tw.x, rtol=0, atol=1e-15)
xo.assert_allclose(tw_back.y, tw.y, rtol=0, atol=1e-15)
xo.assert_allclose(tw_back.zeta, tw.zeta, rtol=0, atol=1e-15)
xo.assert_allclose(tw_back.delta, tw.delta, rtol=0, atol=1e-15)
xo.assert_allclose(tw_back.px, tw.px, rtol=0, atol=1e-15)
xo.assert_allclose(tw_back.py, tw.py, rtol=0, atol=1e-15)
xo.assert_allclose(tw_back.betx1, tw.betx1, rtol=1e-7, atol=1e-15)
xo.assert_allclose(tw_back.bety1, tw.bety1, rtol=1e-7, atol=1e-15)
xo.assert_allclose(tw_back.betx2, tw.betx2, rtol=1e-7, atol=1e-15)
xo.assert_allclose(tw_back.bety2, tw.bety2, rtol=1e-7, atol=1e-15)