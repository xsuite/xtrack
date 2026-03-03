import xtrack as xt

m = xt.Multipole(knl=[1e-3])

p0 = xt.Particles(p0c=1e9)

p = p0.copy()
m.track(p)

m.knl_rel = [0.5]
p = p0.copy()
m.track(p)