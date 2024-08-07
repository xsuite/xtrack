import xtrack as xt
import xobjects as xo

xo.context_default.kernels.clear()

knl = [0.01]
ksl = [0]

quad = xt.Quadrupole(k1=0, length=1, knl=knl, ksl=ksl)
d1 = xt.Drift(length=0.5)
m = xt.Multipole(length=1, knl=knl, ksl=ksl)
d2 = xt.Drift(length=0.5)

l2 = xt.Line(elements=[d1, m, d2])
l2.build_tracker()


p0 = xt.Particles(p0c=7000e9, x=1e-3, px=0)

p1 = p0.copy()
p2 = p0.copy()

quad.track(p1)
l2.track(p2)