import xtrack as xt
import xobjects as xo
import numpy as np

length = 3.
ks = 2.

sol = xt.UniformSolenoid(length=length, ks=ks)
ref_sol = xt.Solenoid(length=length, ks=ks) # Old solenoid


p0 = xt.Particles(p0c=1e9, x=1e-3, y=2e-3)

p = p0.copy()
p_ref = p0.copy()

sol.track(p)
ref_sol.track(p_ref)

xo.assert_allclose(p.x, p_ref.x, rtol=0, atol=1e-10)
xo.assert_allclose(p.y, p_ref.y, rtol=0, atol=1e-10)
xo.assert_allclose(p.px, p_ref.px, rtol=0, atol=1e-10)
xo.assert_allclose(p.py, p_ref.py, rtol=0, atol=1e-10)
xo.assert_allclose(p.delta, p_ref.delta, rtol=0, atol=1e-10)
xo.assert_allclose(p.ax, 0., rtol=0, atol=1e-10)
xo.assert_allclose(p.ay, 0., rtol=0, atol=1e-10)
xo.assert_allclose(p.kin_px, p_ref.px, rtol=0, atol=1e-10)
xo.assert_allclose(p.kin_py, p_ref.py, rtol=0, atol=1e-10)

sol.edge_exit_active = False
p = p0.copy()
sol.track(p)

xo.assert_allclose(p.x, p_ref.x, rtol=0, atol=1e-10)
xo.assert_allclose(p.y, p_ref.y, rtol=0, atol=1e-10)
xo.assert_allclose(p.px, p_ref.px, rtol=0, atol=1e-10)
xo.assert_allclose(p.py, p_ref.py, rtol=0, atol=1e-10)
xo.assert_allclose(p.delta, p_ref.delta, rtol=0, atol=1e-10)
xo.assert_allclose(p.ax, p_ref.ax, rtol=0, atol=1e-10)
xo.assert_allclose(p.ay, p_ref.ay, rtol=0, atol=1e-10)
xo.assert_allclose(p.kin_px, p_ref.kin_px, rtol=0, atol=1e-10)
xo.assert_allclose(p.kin_py, p_ref.kin_py, rtol=0, atol=1e-10)

p0_for_backtrack = p.copy()
p_for_backtrack = p0_for_backtrack.copy()
lsol = xt.Line([sol])

lsol.track(p_for_backtrack, backtrack=True)
xo.assert_allclose(p_for_backtrack.x, p0.x, rtol=0, atol=1e-10)
xo.assert_allclose(p_for_backtrack.y, p0.y, rtol=0, atol=1e-10)
xo.assert_allclose(p_for_backtrack.px, p0.px, rtol=0, atol=1e-10)
xo.assert_allclose(p_for_backtrack.py, p0.py, rtol=0, atol=1e-10)
xo.assert_allclose(p_for_backtrack.delta, p0.delta, rtol=0, atol=1e-10)
xo.assert_allclose(p_for_backtrack.ax, 0., rtol=0, atol=1e-10)
xo.assert_allclose(p_for_backtrack.ay, 0., rtol=0, atol=1e-10)
xo.assert_allclose(p_for_backtrack.kin_px, p0.px, rtol=0, atol=1e-10)
xo.assert_allclose(p_for_backtrack.kin_py, p0.py, rtol=0, atol=1e-10)

sol.edge_entry_active = False
p_for_backtrack = p0_for_backtrack.copy()
lsol.track(p_for_backtrack, backtrack=True)
xo.assert_allclose(p_for_backtrack.x, p0.x, rtol=0, atol=1e-10)
xo.assert_allclose(p_for_backtrack.y, p0.y, rtol=0, atol=1e-10)
xo.assert_allclose(p_for_backtrack.px, p0.px, rtol=0, atol=1e-10)
xo.assert_allclose(p_for_backtrack.py, p0.py, rtol=0, atol=1e-10)
xo.assert_allclose(p_for_backtrack.delta, p0.delta, rtol=0, atol=1e-10)
xo.assert_allclose(p_for_backtrack.ax, -ks /2 *p0.y, rtol=0, atol=1e-10)
xo.assert_allclose(p_for_backtrack.ay, ks /2 *p0.x, rtol=0, atol=1e-10)
xo.assert_allclose(p_for_backtrack.kin_px, p0.px + ks /2 *p0.y, rtol=0, atol=1e-10)
xo.assert_allclose(p_for_backtrack.kin_py, p0.py - ks /2 *p0.x, rtol=0, atol=1e-10)

lsol_sliced = lsol.copy(shallow=True)
lsol_sliced.cut_at_s([length / 3, 2 * length / 3])
tt_sliced = lsol_sliced.get_table(attr=True)

sol.edge_entry_active = True
sol.edge_exit_active = True

assert np.all(tt_sliced.name == np.array(
    ['e0_entry', 'e0..entry_map', 'e0..0', 'e0..1', 'e0..2',
       'e0..exit_map', 'e0_exit', '_end_point']))

xo.assert_allclose(tt_sliced.s, np.array([0., 0., 0., 1., 2., 3., 3., 3.]),
                   rtol=0, atol=1e-10)

assert np.all(tt_sliced.element_type == np.array(
    ['Marker', 'ThinSliceUniformSolenoidEntry',
       'ThickSliceUniformSolenoid', 'ThickSliceUniformSolenoid',
       'ThickSliceUniformSolenoid', 'ThinSliceUniformSolenoidExit',
       'Marker', '']))

lsol_sliced.particle_ref = xt.Particles(p0c=100e9)
tw = lsol_sliced.twiss(x=p0.x, px=p0.px, y=p0.y, py=p0.py, betx=1, bety=1)
p_ref = p0.copy()
ref_sol.track(p_ref)

xo.assert_allclose(tw.x[-1], p_ref.x, rtol=0, atol=1e-10)
xo.assert_allclose(tw.px[-1], p_ref.px, rtol=0, atol=1e-10)
xo.assert_allclose(tw.y[-1], p_ref.y, rtol=0, atol=1e-10)
xo.assert_allclose(tw.py[-1], p_ref.py, rtol=0, atol=1e-10)
xo.assert_allclose(tw.delta[-1], p_ref.delta, rtol=0, atol=1e-10)

tw['ax'] = tw.px - tw.kin_px
tw['ay'] = tw.py - tw.kin_py


xo.assert_allclose(tw.kin_px[-1], p_ref.kin_px, rtol=0, atol=1e-10)
xo.assert_allclose(tw.kin_py[-1], p_ref.kin_py, rtol=0, atol=1e-10)
