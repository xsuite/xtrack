import numpy as np

import xtrack as xt
import xobjects as xo

m = xt.Multipole(knl=[0.1, 0], hxl=0.1, length=2)
p = xt.Particles(x = 0, y=0, delta=1, p0c=1e12)
ln = xt.Line(elements=[
    xt.SRotation(angle=-90.),
    m,
    xt.SRotation(angle=90.)])
ln.build_tracker()
ln.track(p)

# Check dispersion
my = xt.Multipole(ksl=[0.1, 0], hyl=0.1, length=2)
py = xt.Particles(x = 0, y=0, delta=1., p0c=1e12)
my.track(py)

p.move(_context=xo.context_default)

assert np.allclose(p.x, py.x, rtol=0, atol=1e-14)
assert np.allclose(p.y, py.y, rtol=0, atol=1e-14)
assert np.allclose(p.px, py.px, rtol=0, atol=1e-14)
assert np.allclose(p.py, py.py, rtol=0, atol=1e-14)
assert np.allclose(p.zeta, py.zeta, rtol=0, atol=1e-14)
assert np.allclose(p.ptau, py.ptau, rtol=0, atol=1e-14)

# Check weak focusing
pf = xt.Particles(x=0, y=0.3, delta=0., p0c=1e12)
pfy = pf.copy()

ln.track(pf)
my.track(pfy)

pf.move(_context=xo.context_default)
pfy.move(_context=xo.context_default)

assert np.allclose(pf.x, pfy.x, rtol=0, atol=1e-14)
assert np.allclose(pf.y, pfy.y, rtol=0, atol=1e-14)
assert np.allclose(pf.px, pfy.px, rtol=0, atol=1e-14)
assert np.allclose(pf.py, pfy.py, rtol=0, atol=1e-14)
assert np.allclose(pf.zeta, pfy.zeta, rtol=0, atol=1e-14)
assert np.allclose(pf.ptau, pfy.ptau, rtol=0, atol=1e-14)
