import xtrack as xt
import xobjects as xo

env = xt.Environment()

line = env.new_line(name='line', components=[
    env.new('qd', xt.Quadrupole, k1=0.1, length=0.1),
])
line_thick = line.copy(shallow=True)
line.slice_thick_elements(slicing_strategies=[xt.Strategy(slicing=xt.Teapot(3))])

p0 = xt.Particles(kinetic_energy0=100e6, mass0=xt.PROTON_MASS_EV,
                  x = 1e-3, px=1e-3, y=1e-3, py=1e-3, zeta=1e-3, delta=1e-3)

p_ref_no_fringe = p0.copy()
p_slice_no_fringe = p0.copy()

line_thick.track(p_ref_no_fringe)
line.track(p_slice_no_fringe)

line.configure_quadrupole_model(edge='full')
p_ref_fringe = p0.copy()
p_slice_fringe = p0.copy()

line_thick.track(p_ref_fringe)
line.track(p_slice_fringe)

diff_no_fringe_x = p_ref_no_fringe.x - p_slice_no_fringe.x
diff_fringe_x = p_ref_fringe.x - p_slice_fringe.x

xo.assert_allclose(diff_no_fringe_x, diff_fringe_x, rtol=0, atol=1e-16)
