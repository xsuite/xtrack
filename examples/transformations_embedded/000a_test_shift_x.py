import xtrack as xt
import numpy as np


k1 = 2.
length = 0.1

quad = xt.Quadrupole(k1=k1, length=length, shift_x=1e-3)

assert quad.shift_x == 1e-3
assert quad.shift_y == 0
assert quad._sin_rot_s == 0.0
assert quad._cos_rot_s == 1.0

p = xt.Particles(x=0, p0c=1e12)
quad.track(p)

assert_allclose = np.testing.assert_allclose
assert_allclose(p.px, -k1 * length * 1e-3, rtol=0, atol=1e-3)

# Change the shift
quad.shift_x = 2e-3
p = xt.Particles(x=0, p0c=1e12)
quad.track(p)
assert_allclose(p.px, -k1 * length*2e-3, rtol=0, atol=1e-3)