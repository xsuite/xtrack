import xtrack as xt
import numpy as np

k1 = 1.0
k1s = 2.0

length = 0.5

quad = xt.Quadrupole(k1=k1, k1s=k1s, length=length)

n_slices = 1000
ele_thin = []
for ii in range(n_slices):
    ele_thin.append(xt.Drift(length=length/n_slices/2))
    ele_thin.append(xt.Multipole(knl=[0, k1 * length/n_slices],
                                 ksl=[0, k1s * length/n_slices]))
    ele_thin.append(xt.Drift(length=length/n_slices/2))
lref = xt.Line(ele_thin)
lref.build_tracker()

p_test = xt.Particles(gamma0=1.2, x=0.1, y=0.2, delta=0.5)
p_ref = p_test.copy()

quad.track(p_test)
lref.track(p_ref)

assert np.isclose(p_test.x, p_ref.x, atol=1e-8, rtol=0)
assert np.isclose(p_test.px, p_ref.px, atol=5e-8, rtol=0)
assert np.isclose(p_test.y, p_ref.y, atol=1e-8, rtol=0)
assert np.isclose(p_test.py, p_ref.py, atol=5e-8, rtol=0)
assert np.isclose(p_test.zeta, p_ref.zeta, atol=1e-8, rtol=0)
assert np.isclose(p_test.delta, p_ref.delta, atol=5e-8, rtol=0)