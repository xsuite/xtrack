import xtrack as xt

angle = 0.5
b2 = 5
b1 = 0

dipole = xt.Bend(length=0, k0=b1, knl=[0, b2], edge_exit_angle=angle, edge_exit_model='2')
sextupole = xt.Multipole(knl = [0, 0, 3/4 * b2 * angle])
line = xt.Line(elements=[dipole, sextupole])

dipole_full = xt.Bend(length=0, k0=b1, knl=[0, b2], edge_exit_angle=angle, edge_exit_model='1')

p0 = xt.Particles(x=0.01, y=0.01)
p1 = p0.copy()

line.track(p0)
print(p0.x, p0.px, p0.y, p0.py)
dipole_full.track(p1)
print(p1.x, p1.px, p1.y, p1.py)