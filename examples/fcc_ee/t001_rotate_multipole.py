import xtrack as xt

m = xt.Multipole(knl=[0.1, 0], hxl=0.1, length=2)
p = xt.Particles(x = 0, y=0, delta=1, p0c=1e12)
ln = xt.Line(elements=[
    xt.SRotation(angle=-90.),
    m,
    xt.SRotation(angle=90.)])
ln.build_tracker()
ln.track(p)

my = xt.Multipole(ksl=[0.1, 0], hyl=0.1, length=2)
py = xt.Particles(x = 0, y=0, delta=1., p0c=1e12)
my.track(py)

pf = xt.Particles(x=0, y=0.3, delta=0., p0c=1e12)
pfy = pf.copy()

ln.track(pf)
my.track(pfy)