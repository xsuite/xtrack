import xtrack as xt
import numpy as np

k3 = 1.0
k3s = 2.0

length = 0.5

oct = xt.Octupole(k3=k3, k3s=k3s, length=length)

ele_thin = []
ele_thin.append(xt.Drift(length=length/2))
ele_thin.append(xt.Multipole(knl=[0, 0, 0, k3 * length],
                             ksl=[0, 0, 0, k3s * length]))
ele_thin.append(xt.Drift(length=length/2))
lref = xt.Line(ele_thin)
lref.build_tracker()

p_test = xt.Particles(gamma0=1.2, x=0.1, y=0.2, delta=0.5)
p_ref = p_test.copy()

oct.track(p_test)
lref.track(p_ref)

assert np.isclose(p_test.x, p_ref.x, atol=1e-12, rtol=0)
assert np.isclose(p_test.px, p_ref.px, atol=1e-12, rtol=0)
assert np.isclose(p_test.y, p_ref.y, atol=1e-12, rtol=0)
assert np.isclose(p_test.py, p_ref.py, atol=1e-12, rtol=0)
assert np.isclose(p_test.zeta, p_ref.zeta, atol=1e-12, rtol=0)
assert np.isclose(p_test.delta, p_ref.delta, atol=1e-12, rtol=0)