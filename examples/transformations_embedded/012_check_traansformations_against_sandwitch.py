import xtrack as xt
import numpy as np

shift_x = 1e-3
shift_y = 2e-3
rot_s_rad = -0.4

elem = xt.Quadrupole(k1=2., k1s=-3., length=3.)

line_test = xt.Line(elements=[elem.copy()])

line_ref = xt.Line(elements=[
    xt.XYShift(dx=shift_x, dy=shift_y),
    xt.SRotation(angle=np.rad2deg(rot_s_rad)),
    elem.copy(),
    xt.SRotation(angle=np.rad2deg(-rot_s_rad)),
    xt.XYShift(dx=-shift_x, dy=-shift_y),
])

line_test['e0'].rot_s_rad = rot_s_rad
line_test['e0'].shift_x = shift_x
line_test['e0'].shift_y = shift_y

p_test = xt.Particles(p0c=10e9, x=0.1, px=0.2, y=0.3, py=0.4, delta=0.03)
p_ref = p_test.copy()

line_test.build_tracker()
line_ref.build_tracker()

line_test.track(p_test)
line_ref.track(p_ref)

assert_allclose = np.testing.assert_allclose
assert_allclose(p_test.x, p_ref.x, rtol=0, atol=1e-13)
assert_allclose(p_test.px, p_ref.px, rtol=0, atol=1e-13)
assert_allclose(p_test.y, p_ref.y, rtol=0, atol=1e-13)
assert_allclose(p_test.py, p_ref.py, rtol=0, atol=1e-13)
assert_allclose(p_test.zeta, p_ref.zeta, rtol=0, atol=1e-13)
assert_allclose(p_test.delta, p_ref.delta, rtol=0, atol=1e-13)

